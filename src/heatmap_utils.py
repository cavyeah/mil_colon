"""
heatmap_utils.py
================
Utility module for:
  1. Extracting FULL (non-top-K) attention scores from a CLAM model
  2. Building a 2-D attention grid from patch coordinates
  3. Plotting and saving heatmaps
  4. ROC curve + Confusion matrix visualisation
  5. Multi-fold attention ensemble averaging
  6. Spatial heatmap + overlay generation (pixel-accurate, patch-stamped)

Target overlay appearance
-------------------------
  - Each 256×256 patch is individually coloured by JET(attention_value)
  - Background (no-patch pixels) = pure black → renders as dark navy after blend
  - Tissue is visible underneath via addWeighted blend
  - Result: tissue shows through with per-patch JET colour tinting
  - Background: RGB ≈ (0, 0, 77) at alpha=0.6  (= 0.6 × JET(0))
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc as sklearn_auc, confusion_matrix
import cv2


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _to_numpy(array) -> np.ndarray:
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _normalize01(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    mn, mx = float(np.min(values)), float(np.max(values))
    if mx > mn:
        return (values - mn) / (mx - mn)
    return np.zeros_like(values, dtype=np.float32)


def _prepare_coords(coords, patch_size: int):
    """
    Return (x, y) float32 arrays in pixel space.
    If coords look like grid indices (max < 5000) they are multiplied by patch_size.
    """
    coords_np = _to_numpy(coords)
    if coords_np.ndim != 2 or coords_np.shape[1] < 2:
        raise ValueError(f"coords must have shape [N, 2], got {coords_np.shape}")

    coords_np = coords_np[:, :2].astype(np.float32, copy=False)
    x = coords_np[:, 0]
    y = coords_np[:, 1]

    if float(np.max(x)) < 5000.0 and float(np.max(y)) < 5000.0:
        x = x * float(patch_size)
        y = y * float(patch_size)

    return x, y


# ──────────────────────────────────────────────────────────────────────────────
# 1.  FULL ATTENTION EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def get_full_attention(model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    """
    Extract normalised attention scores for ALL N patches — no Top-K filtering.

    Returns
    -------
    A_full : torch.Tensor of shape [N, 1]
        Softmax-normalised attention weight for every patch.
    """
    model.eval()
    with torch.no_grad():
        raw_A = model.attention_net.attention(features)   # [N, 1]
        raw_A = raw_A - raw_A.max()                       # numerical stability
        A_full = F.softmax(raw_A, dim=0)                  # [N, 1]
    return A_full


# ──────────────────────────────────────────────────────────────────────────────
# 2.  GRID RECONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

def create_attention_grid(
    coords,
    attention,
    grid_size=8,
) -> np.ndarray:
    """
    Aggregate patch attention scores onto a regular 2-D grid.

    Parameters
    ----------
    coords    : [N, 2] — (x, y) pixel coordinates
    attention : [N] or [N, 1] — attention scores
    grid_size : int or (rows, cols)

    Returns
    -------
    grid_norm : np.ndarray [grid_rows, grid_cols], values in [0, 1]
    """
    coords_np = _to_numpy(coords)
    attn_np   = _to_numpy(attention).flatten()

    x, y = _prepare_coords(coords_np, patch_size=256)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    x_norm = (x - x_min) / x_range
    y_norm = (y - y_min) / y_range

    if isinstance(grid_size, int):
        grid_rows, grid_cols = grid_size, grid_size
    else:
        grid_rows, grid_cols = int(grid_size[0]), int(grid_size[1])

    N = len(attn_np)
    row_idx = (y_norm * (grid_rows - 1e-9)).astype(int)
    col_idx = (x_norm * (grid_cols - 1e-9)).astype(int)
    row_idx = np.clip(row_idx, 0, grid_rows - 1)
    col_idx = np.clip(col_idx, 0, grid_cols - 1)

    grid   = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    counts = np.zeros((grid_rows, grid_cols), dtype=np.float64)

    for r, c, a in zip(row_idx, col_idx, attn_np):
        grid[r, c]   += a
        counts[r, c] += 1

    with np.errstate(invalid="ignore"):
        grid_mean = np.where(counts > 0, grid / counts, 0.0)

    g_min, g_max = grid_mean.min(), grid_mean.max()
    if g_max > g_min:
        grid_norm = (grid_mean - g_min) / (g_max - g_min)
    else:
        grid_norm = np.zeros_like(grid_mean)

    grid_norm = np.power(grid_norm, 0.5)   # gamma boost for mid-values
    return grid_norm


# ──────────────────────────────────────────────────────────────────────────────
# 3.  SPATIAL HEATMAP  (standalone — no tissue image)
# ──────────────────────────────────────────────────────────────────────────────

def create_spatial_heatmap(
    coords,
    attention,
    patch_size: int = 256,
    threshold:  float = 0.0,
    alpha:      float = 0.6,           # kept for API compat; ignored in standalone mode
    smoothing:  bool = False,          # light smoothing only
    smoothing_sigma: float | None = None,
    original_image: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate a pixel-accurate spatial attention heatmap.

    Standalone mode  (original_image=None)
    ----------------------------------------
    Returns a JET-coloured heatmap on a BLACK canvas.
    Each patch cell is filled with the JET colour corresponding to its
    normalised attention value.  Background (no-patch) pixels stay black.

    Overlay mode  (original_image provided)
    ----------------------------------------
    Delegates to create_overlay_heatmap() for a proper blended result.

    Returns
    -------
    np.ndarray [H, W, 3] uint8
    """
    if original_image is not None:
        return create_overlay_heatmap(
            coords=coords,
            attention=attention,
            original_image=original_image,
            patch_size=patch_size,
            alpha=alpha,
            smoothing=smoothing,
            smoothing_sigma=smoothing_sigma,
        )

    # ── standalone heatmap ────────────────────────────────────────────────────
    attn_np = _to_numpy(attention).astype(np.float32).flatten()
    x, y    = _prepare_coords(coords, patch_size)

    # Shift so top-left is (0,0)
    x = x - float(np.min(x))
    y = y - float(np.min(y))

    # Canvas size capped at 4096 to avoid memory issues
    canvas_w = int(np.max(x) + patch_size)
    canvas_h = int(np.max(y) + patch_size)
    max_dim   = 4096
    scale = min(1.0, max_dim / float(max(canvas_h, canvas_w, 1)))
    if scale < 1.0:
        x          = x * scale
        y          = y * scale
        patch_size = max(1, int(round(patch_size * scale)))
        canvas_w   = int(np.max(x) + patch_size)
        canvas_h   = int(np.max(y) + patch_size)

    canvas_h = max(1, canvas_h)
    canvas_w = max(1, canvas_w)

    # Normalise attention → [0, 1]
    attn_norm = _normalize01(attn_np)

    # Optional contrast boost
    attn_norm = np.power(attn_norm, 0.6)

    # Threshold
    if threshold > 0.0:
        attn_norm[attn_norm < threshold] = 0.0

    # Build float heatmap canvas (0 = no patch / background)
    heatmap = np.full((canvas_h, canvas_w), -1.0, dtype=np.float32)

    for px, py, val in zip(x.astype(int), y.astype(int), attn_norm):
        x0 = max(0, px);       y0 = max(0, py)
        x1 = min(canvas_w, x0 + patch_size)
        y1 = min(canvas_h, y0 + patch_size)
        if x1 <= x0 or y1 <= y0:
            continue
        heatmap[y0:y1, x0:x1] = val

    # Create mask of actual patch pixels
    patch_mask = (heatmap >= 0.0)

    # Set background to 0 for colormap application
    heatmap[~patch_mask] = 0.0

    # Optional very light smoothing (patch-aware)
    if smoothing:
        sigma = smoothing_sigma if smoothing_sigma is not None else 1.0
        k     = max(3, int(round(sigma * 4)) | 1)
        k     = min(k, 31)
        heatmap = cv2.GaussianBlur(heatmap, (k, k), sigma)

    # Apply JET colormap
    heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)   # BGR
    heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Black out background pixels
    heatmap_rgb[~patch_mask] = 0

    return heatmap_rgb


# ──────────────────────────────────────────────────────────────────────────────
# 4.  OVERLAY HEATMAP  (pixel-accurate patch stamp — matches target image)
# ──────────────────────────────────────────────────────────────────────────────

def create_overlay_heatmap(
    coords,
    attention,
    original_image: np.ndarray,
    patch_size: int = 256,
    alpha: float = 0.6,
    smoothing: bool = False,
    smoothing_sigma: float | None = None,
) -> np.ndarray:
    """
    Blend a per-patch JET attention heatmap onto the original WSI thumbnail.

    Algorithm
    ---------
    1.  Normalise attention values to [0, 1].
    2.  Stamp each 256×256 patch onto a float heatmap canvas at its pixel coord.
    3.  Apply JET colourmap to the entire canvas.
    4.  Set background (no-patch) pixels to black on both heatmap AND tissue.
    5.  Blend: result = alpha × heatmap_rgb + (1 − alpha) × tissue_rgb
        Background pixels stay: alpha × JET(0) + (1−alpha) × black
                               ≈ (0, 0, round(128 × alpha))  in RGB
                               ≈ (0, 0, 77) at alpha=0.6  ✓ matches target

    Parameters
    ----------
    coords         : [N, 2] pixel coordinates (unscaled, original resolution)
    attention      : [N] or [N, 1] normalised attention scores
    original_image : [H, W, 3] uint8 RGB tissue thumbnail
    patch_size     : patch side length in pixels (default 256)
    alpha          : heatmap opacity (0 = tissue only, 1 = heatmap only)
    smoothing      : apply very light Gaussian smoothing to heatmap
    smoothing_sigma: sigma for Gaussian kernel

    Returns
    -------
    overlay : [H, W, 3] uint8 RGB
    """
    attn_np = _to_numpy(attention).astype(np.float32).flatten()
    tissue_h, tissue_w = original_image.shape[:2]

    # ── get pixel coords from feature bag ────────────────────────────────────
    x_raw, y_raw = _prepare_coords(coords, patch_size)

    # ── figure out the native WSI extent (from patch coords) ─────────────────
    x_min_raw = float(np.min(x_raw))
    y_min_raw = float(np.min(y_raw))
    x_max_raw = float(np.max(x_raw))
    y_max_raw = float(np.max(y_raw))

    native_w = x_max_raw - x_min_raw + patch_size   # full WSI width at feature resolution
    native_h = y_max_raw - y_min_raw + patch_size   # full WSI height

    # ── scale coords so they span the tissue thumbnail exactly ────────────────
    scale_x = tissue_w / native_w
    scale_y = tissue_h / native_h
    # Use the smaller scale to preserve aspect ratio, then centre
    scale   = min(scale_x, scale_y)

    x_scaled = (x_raw - x_min_raw) * scale
    y_scaled = (y_raw - y_min_raw) * scale
    ps_scaled = max(1, int(round(patch_size * scale)))

    # Offset to centre within tissue thumbnail
    total_w_scaled = (x_max_raw - x_min_raw) * scale + ps_scaled
    total_h_scaled = (y_max_raw - y_min_raw) * scale + ps_scaled
    off_x = int((tissue_w - total_w_scaled) / 2)
    off_y = int((tissue_h - total_h_scaled) / 2)

    x_final = (x_scaled + off_x).astype(int)
    y_final = (y_scaled + off_y).astype(int)

    # ── normalise attention ───────────────────────────────────────────────────
    attn_norm = _normalize01(attn_np)
    attn_norm = np.power(attn_norm, 0.6)   # mild contrast boost

    # ── build float heatmap canvas ────────────────────────────────────────────
    # -1.0 marks "no patch" (background)
    heatmap = np.full((tissue_h, tissue_w), -1.0, dtype=np.float32)

    for px, py, val in zip(x_final, y_final, attn_norm):
        x0 = max(0, int(px));          y0 = max(0, int(py))
        x1 = min(tissue_w, x0 + ps_scaled)
        y1 = min(tissue_h, y0 + ps_scaled)
        if x1 <= x0 or y1 <= y0:
            continue
        # Only overwrite if this pixel hasn't been set yet, or take max
        region = heatmap[y0:y1, x0:x1]
        heatmap[y0:y1, x0:x1] = np.maximum(region, val)

    patch_mask = (heatmap >= 0.0)   # True where a patch was stamped

    # Fill background with 0 before colourmap
    heatmap[~patch_mask] = 0.0

    # ── optional light smoothing ──────────────────────────────────────────────
    if smoothing:
        sigma = smoothing_sigma if smoothing_sigma is not None else max(1.0, ps_scaled / 8.0)
        k     = max(3, int(round(sigma * 4)) | 1)
        k     = min(k, 31)
        heatmap_smooth = cv2.GaussianBlur(heatmap, (k, k), sigma)
        # Only apply smoothing inside patch area to preserve sharp borders
        heatmap[patch_mask] = heatmap_smooth[patch_mask]

    # ── apply JET colourmap ───────────────────────────────────────────────────
    heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    heatmap_bgr   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb   = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Background stays as JET(0) = (0, 0, 128) in RGB
    # After blend with black tissue: alpha×(0,0,128) = (0,0,77) at alpha=0.6

    # ── prepare tissue ────────────────────────────────────────────────────────
    tissue = original_image.copy().astype(np.float32)
    # Pixels outside patch area: set tissue to black
    # (background will be purely the JET heatmap contribution)
    tissue[~patch_mask] = 0.0

    # ── blend ────────────────────────────────────────────────────────────────
    heatmap_f = heatmap_rgb.astype(np.float32)
    overlay   = alpha * heatmap_f + (1.0 - alpha) * tissue
    overlay   = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


# ──────────────────────────────────────────────────────────────────────────────
# COMPATIBILITY WRAPPER
# ──────────────────────────────────────────────────────────────────────────────

def build_attention_overlay(
    coords,
    attention,
    original_image: np.ndarray,
    patch_size: int = 256,
    alpha: float = 0.6,
    threshold: float = 0.0,
    smoothing: bool = False,
    smoothing_sigma: float | None = None,
) -> np.ndarray:
    """Compatibility wrapper — delegates to create_overlay_heatmap."""
    return create_overlay_heatmap(
        coords=coords,
        attention=attention,
        original_image=original_image,
        patch_size=patch_size,
        alpha=alpha,
        smoothing=smoothing,
        smoothing_sigma=smoothing_sigma,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5.  HEATMAP VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_heatmap(
    grid:        np.ndarray,
    title:       str = "Attention Heatmap",
    output_path: str | None = None,
    cmap:        str = "jet",
    figsize:     tuple = (8, 6),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid, cmap=cmap, interpolation="nearest",
                   origin="upper", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalised Attention", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Grid column  (→ x)", fontsize=10)
    ax.set_ylabel("Grid row  (→ y)", fontsize=10)
    g = grid.shape[0]
    ax.set_xticks(range(g))
    ax.set_yticks(range(g))
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  ROC CURVE
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    labels:      np.ndarray,
    probs:       np.ndarray,
    output_path: str = "roc_curve.png",
    threshold:   float | None = None,
) -> None:
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = sklearn_auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2,
            label=f"ROC curve  (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random")
    if threshold is not None:
        idx = np.argmin(np.abs(thresholds - threshold))
        ax.scatter(fpr[idx], tpr[idx], s=80, zorder=5, color="#F44336",
                   label=f"Threshold = {threshold:.3f}")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  CONFUSION MATRIX
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels:      np.ndarray,
    preds:       np.ndarray,
    class_names: list = ["Normal", "Tumor"],
    output_path: str  = "confusion_matrix.png",
) -> None:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks);  ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticks(tick_marks);  ax.set_yticklabels(class_names, fontsize=10)
    total  = cm.sum()
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct   = 100 * count / total
            color = "white" if count > thresh else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", color=color, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  MULTI-FOLD ATTENTION ENSEMBLE
# ──────────────────────────────────────────────────────────────────────────────

def ensemble_attention(
    model_class,
    model_paths: list,
    features:    torch.Tensor,
    device      = "cpu",
) -> torch.Tensor:
    """Average full attention scores from multiple fold models."""
    if not model_paths:
        raise ValueError("model_paths must not be empty")

    device   = torch.device(device) if isinstance(device, str) else device
    features = features.float().to(device)
    A_list   = []

    for path in model_paths:
        if not os.path.exists(path):
            print(f"  [WARN] checkpoint not found, skipping: {path}")
            continue
        model = model_class().to(device)
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif hasattr(state, "state_dict"):
            state = state.state_dict()
        clean = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(clean, strict=False)
        A_list.append(get_full_attention(model, features))   # [N, 1]

    if not A_list:
        raise RuntimeError("No valid checkpoints found in model_paths")

    return torch.stack(A_list, dim=0).mean(dim=0)   # [N, 1]
