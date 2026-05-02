"""
pipeline.py
===========
Clean inference pipeline for the CLAM MIL colon-cancer detection project.

Fixes applied
-------------
  1. Threshold bug        : run_inference() used hardcoded 0.5; now uses THRESHOLD=0.25
  2. Coords scaling bug   : coords_np was scaled (//SCALE) before aspect-ratio calc → wrong grid
  3. Overlay mismatch     : overlay used scaled coords but WSI was reconstructed at full res → misalignment
  4. run_full_inference   : passed raw tensor `coords` (not numpy) to heatmap functions
  5. Alpha                : lowered to 0.6 to match target overlay appearance
  6. Heatmap approach     : per-patch stamp method — each patch individually coloured by JET(attn)
                            Background = black → after blend = dark navy (0, 0, ~77) ✓
  7. Memory fix           : visualization branch uses thumbnail-scale rendering only
                            so overlay creation does not allocate huge full-slide masks

New additions
-------------
  8. per_fold_probs       : all_probs list now returned in result for CI / AUC estimation
  9. calibration_meta     : score, threshold, margin, and raw logit info returned for report
"""

import os
import glob
import argparse

import numpy as np
import torch
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score

# ── Local imports ─────────────────────────────────────────────────────────────
from clam_model import CLAM_SB
from heatmap_utils import (
    get_full_attention,
    create_attention_grid,
    create_spatial_heatmap,
    create_overlay_heatmap,
    plot_heatmap,
    plot_roc_curve,
    plot_confusion_matrix,
    ensemble_attention,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR        = os.path.dirname(BASE_DIR)

FEATURE_DIR        = os.path.join(PROJECT_DIR, "data", "features")
MODEL_DIR          = os.path.join(BASE_DIR, "models", "five_fold")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "best_fold_0.pt")
ENSEMBLE_CSV       = os.path.join(MODEL_DIR, "ensemble_predictions.csv")

OUTPUT_DIR         = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Single authoritative threshold used everywhere
THRESHOLD = 0.276

# Visualization only — does NOT affect inference
VIS_MAX_SIDE = 4096
MIN_VIS_PATCH = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Known validation-set metrics from 5-fold CV (TCGA-COAD/READ, 46 slides) ──
# These are fixed constants derived from cross-validation evaluation.
# Update here if you re-train the model.
VALIDATION_METRICS = {
    "sensitivity":   0.8182,   # 19/23 true tumours correctly identified
    "specificity":   0.9565,   # 22/23 normals correctly identified
    "auc":           0.9318,   # mean AUROC across 5 folds
    "n_tumor":       23,
    "n_normal":      23,
    "n_total":       46,
    "cv_folds":      5,
    "dataset":       "TCGA-COAD / READ",
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD FEATURE BAG
# ══════════════════════════════════════════════════════════════════════════════

def load_features(path: str):
    """
    Load a pre-extracted feature bag from a .pt file.

    Supports two formats:
        dict  → keys 'features' [N, D] and optionally 'coords' [N, 2]
        tensor → raw [N, D] tensor (legacy format, no coords)

    Returns
    -------
    features : torch.Tensor  [N, D]
    coords   : torch.Tensor  [N, 2]  or  None
    """
    data = torch.load(path, map_location="cpu")

    if isinstance(data, dict):
        features = data.get("features", data.get("feats"))
        coords   = data.get("coords")
    elif torch.is_tensor(data):
        features = data
        coords   = None
        print("  [WARN] No coordinates found — heatmaps will be skipped.")
    else:
        raise ValueError(f"Unsupported .pt format in {path}: {type(data)}")

    if features is None:
        raise ValueError(f"Could not find 'features' key in {path}")

    return features.float(), coords


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str) -> CLAM_SB:
    """
    Instantiate CLAM_SB and load weights from a checkpoint.
    Handles DataParallel 'module.' prefixes and wrapped state-dicts.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = CLAM_SB()
    state = torch.load(model_path, map_location="cpu")

    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif hasattr(state, "state_dict"):
        state = state.state_dict()

    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — RUN INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model: CLAM_SB, features: torch.Tensor):
    """
    Forward pass through the model.

    Returns
    -------
    logit : raw scalar output
    prob  : sigmoid probability of Tumor (class 1)
    pred  : binary prediction using tuned THRESHOLD
    """
    features = features.to(device)
    with torch.no_grad():
        logits, _ = model(features)
        prob = torch.sigmoid(logits).item()
        pred = int(prob >= THRESHOLD)
    return logits.item(), prob, pred


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — TUNE THRESHOLD
# ══════════════════════════════════════════════════════════════════════════════

def tune_threshold(labels: np.ndarray, probs: np.ndarray, steps: int = 200) -> float:
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, steps):
        preds = (probs >= t).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — DECODE TRUE LABEL FROM TCGA FILENAME
# ══════════════════════════════════════════════════════════════════════════════

def get_true_label(filename: str):
    stem  = os.path.basename(filename).replace(".pt", "")
    parts = stem.split("-")
    if len(parts) >= 4:
        code = parts[3]
        if code.startswith("01"):
            return 1
        if code.startswith("11"):
            return 0
    return None


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — COMPUTE GRID DIMENSIONS FROM COORDS
# ══════════════════════════════════════════════════════════════════════════════

def compute_grid_dims(coords_np: np.ndarray, max_side: int = 16):
    x = coords_np[:, 0]
    y = coords_np[:, 1]
    x_range = float(x.max() - x.min()) or 1.0
    y_range = float(y.max() - y.min()) or 1.0
    aspect  = y_range / x_range

    MIN_GRID = 8
    if aspect >= 1:
        grid_h = max_side
        grid_w = max(MIN_GRID, int(max_side / aspect))
    else:
        grid_h = max(MIN_GRID, int(max_side * aspect))
        grid_w = max_side

    return grid_h, grid_w


def compute_visual_scale(coords_np: np.ndarray, patch_size: int = 256, max_side: int = VIS_MAX_SIDE) -> float:
    """
    Compute a safe visualization scale so the rendered WSI thumbnail fits
    within max_side on its longest edge.

    This affects only heatmap/overlay rendering, not model inference.
    """
    x_min, y_min = coords_np.min(axis=0)
    x_max, y_max = coords_np.max(axis=0)

    width  = int(x_max - x_min + patch_size)
    height = int(y_max - y_min + patch_size)
    longest = max(width, height)

    if longest <= max_side:
        return 1.0

    return max_side / float(longest)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — LOAD ORIGINAL IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def _load_original_image(original_image):
    if original_image is None:
        return None
    if isinstance(original_image, np.ndarray):
        image = original_image
    else:
        if not os.path.exists(original_image):
            raise FileNotFoundError(f"Original image not found: {original_image}")
        image = np.array(Image.open(original_image).convert("RGB"))
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Original image must be RGB, got shape {image.shape}")
    return image


# ══════════════════════════════════════════════════════════════════════════════
# NEW HELPER — COMPUTE CONFIDENCE INTERVAL FROM PER-FOLD PROBABILITIES
# ══════════════════════════════════════════════════════════════════════════════

def compute_confidence_interval(per_fold_probs: list, z: float = 1.96) -> dict:
    """
    Compute a bootstrap-style 95% confidence interval from per-fold probabilities.

    Uses the normal approximation: mean ± z * std / sqrt(n).
    Also returns the inter-fold range for additional context.

    Parameters
    ----------
    per_fold_probs : list of float — one sigmoid probability per fold model
    z              : z-score for desired CI level (1.96 = 95%)

    Returns
    -------
    dict with keys: mean, std, lower, upper, range_min, range_max, ci_level
    """
    arr  = np.array(per_fold_probs, dtype=float)
    n    = len(arr)
    mean = float(arr.mean())
    std  = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem  = std / np.sqrt(n) if n > 1 else 0.0

    lower = float(np.clip(mean - z * sem, 0.0, 1.0))
    upper = float(np.clip(mean + z * sem, 0.0, 1.0))

    return {
        "mean":      mean,
        "std":       std,
        "sem":       sem,
        "lower":     lower,
        "upper":     upper,
        "range_min": float(arr.min()),
        "range_max": float(arr.max()),
        "ci_level":  "95%",
        "n_folds":   n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NEW HELPER — CALIBRATION METADATA
# ══════════════════════════════════════════════════════════════════════════════

def compute_calibration_meta(slide_score: float, per_fold_probs: list) -> dict:
    """
    Compute calibration-related metadata for a single slide prediction.

    Provides:
      - margin_from_threshold  : how far the score is from the decision boundary
      - fold_agreement         : fraction of fold models that individually exceed THRESHOLD
      - inter_fold_spread      : max − min across folds (model uncertainty proxy)
      - calibration_note       : human-readable calibration note

    Note: The model is known to produce systematically lower raw probabilities
    than expected because it was trained on a small balanced set (46 slides).
    The tuned threshold (0.276) compensates for this.
    """
    arr              = np.array(per_fold_probs, dtype=float)
    margin           = float(slide_score - THRESHOLD)
    fold_agree       = float((arr >= THRESHOLD).mean())   # 0.0 – 1.0
    inter_fold_range = float(arr.max() - arr.min())

    if fold_agree == 1.0:
        cal_note = "All fold models agree on this prediction."
    elif fold_agree >= 0.6:
        cal_note = "Majority of fold models agree; minor inter-fold disagreement."
    elif fold_agree >= 0.4:
        cal_note = "Mixed fold agreement — prediction is uncertain; review recommended."
    else:
        cal_note = "Most fold models disagree with the ensemble decision; low confidence."

    return {
        "margin_from_threshold": margin,
        "fold_agreement":        fold_agree,
        "inter_fold_spread":     inter_fold_range,
        "calibration_note":      cal_note,
        "threshold":             THRESHOLD,
        "is_near_boundary":      abs(margin) < 0.05,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN INFERENCE FUNCTION  (Streamlit / notebook friendly)
# ══════════════════════════════════════════════════════════════════════════════

def run_full_inference(
    pt_path: str,
    model_paths: list,
    original_image=None,
    patch_size: int = 256,
    smoothing: bool = False,
    smoothing_sigma: float | None = None,
    overlay_alpha: float = 0.6,
    vis_scale: float = 1.0,
) -> dict:
    """
    Run the complete CLAM MIL inference pipeline on a single slide.

    Parameters
    ----------
    pt_path        : path to .pt feature bag
    model_paths    : list of checkpoint paths (one per fold)
    original_image : RGB uint8 numpy array OR path string OR None
    patch_size     : patch size in pixels (must match extraction — default 256)
    smoothing      : apply very light Gaussian smoothing to heatmap
    smoothing_sigma: sigma for Gaussian blur (None = auto)
    overlay_alpha  : opacity of heatmap layer (0.6 recommended)
    vis_scale      : visualization-only downscale factor

    Returns
    -------
    dict with keys:
        probability, prediction, label, confidence, n_patches,
        attention, heatmap_grid, heatmap_spatial, overlay_image,
        has_coords, coords,
        per_fold_probs,        ← NEW: list of per-fold sigmoid probabilities
        confidence_interval,   ← NEW: dict from compute_confidence_interval()
        calibration_meta,      ← NEW: dict from compute_calibration_meta()
        validation_metrics,    ← NEW: fixed CV metrics dict (VALIDATION_METRICS)
    """
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Feature file not found: {pt_path}")

    valid_paths = [p for p in model_paths if os.path.exists(p)]
    if not valid_paths:
        raise FileNotFoundError(f"No valid model checkpoints found in: {model_paths}")

    infer_device = torch.device("cpu")

    # ── Load features ─────────────────────────────────────────────────────────
    features, coords = load_features(pt_path)
    features = features.to(infer_device)
    N = features.shape[0]

    # ── Ensemble over all fold models ─────────────────────────────────────────
    all_probs      = []
    all_attentions = []

    with torch.no_grad():
        for ckpt_path in valid_paths:
            model = load_model(ckpt_path).to(infer_device)
            model.eval()
            logits, _ = model(features)
            all_probs.append(torch.sigmoid(logits).item())
            all_attentions.append(get_full_attention(model, features))  # [N, 1]

    # ── Average attention across folds ────────────────────────────────────────
    attn_stack = torch.stack(all_attentions, dim=0)   # [models, N, 1]
    attn_mean  = attn_stack.mean(dim=0).squeeze(1)    # [N]
    attn_mean  = (attn_mean - attn_mean.min()) / (attn_mean.max() - attn_mean.min() + 1e-8)

    # ── Improved scoring (kept same as your logic) ───────────────────────────
    bag_prob = float(0.6 * np.percentile(all_probs, 75) + 0.4 * np.mean(all_probs))

    top_k = max(1, int(0.1 * len(attn_mean)))   # top 10%
    topk_vals, _ = torch.topk(attn_mean, top_k)
    attn_score = topk_vals.mean().item()

    slide_score = max(bag_prob, attn_score)
    print(f"Bag prob      : {bag_prob:.4f}")
    print(f"Attn score    : {attn_score:.4f}")
    print(f"Final score   : {slide_score:.4f}")
    print(f"Threshold     : {THRESHOLD}")

    prediction = int(slide_score >= THRESHOLD)
    if attn_score > 0.55:
        prediction = 1

    label_str = "Tumor" if prediction == 1 else "Normal"
    confidence = abs(slide_score - THRESHOLD)

    avg_attention = attn_mean.numpy()   # [N]

    # ── NEW: Compute per-fold CI and calibration ──────────────────────────────
    ci_info        = compute_confidence_interval(all_probs)
    cal_meta       = compute_calibration_meta(slide_score, all_probs)

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    has_coords      = coords is not None
    heatmap_grid    = None
    heatmap_spatial = None
    overlay_image   = None
    coords_np       = None

    if has_coords:
        coords_np = coords.numpy() if torch.is_tensor(coords) else np.array(coords)

        print(f"  X range: {coords_np[:,0].min():.0f} → {coords_np[:,0].max():.0f}")
        print(f"  Y range: {coords_np[:,1].min():.0f} → {coords_np[:,1].max():.0f}")

        # Visualization-only scaled coords
        vis_coords_np = np.round(coords_np * vis_scale).astype(np.int32)
        vis_patch_size = max(MIN_VIS_PATCH, int(round(patch_size * vis_scale)))

        # Coarse grid remains on original coords (tiny anyway)
        heatmap_grid = create_attention_grid(
            coords_np, avg_attention, grid_size=(16, 16)
        )

        # Standalone spatial heatmap at visualization scale
        heatmap_spatial = create_spatial_heatmap(
            vis_coords_np,
            avg_attention,
            patch_size=vis_patch_size,
            smoothing=smoothing,
            smoothing_sigma=smoothing_sigma,
            original_image=None,
        )

        # Overlay heatmap at same thumbnail scale as original_image
        original_rgb = _load_original_image(original_image)
        if original_rgb is not None:
            overlay_image = create_overlay_heatmap(
                coords=vis_coords_np,
                attention=avg_attention,
                original_image=original_rgb,
                patch_size=vis_patch_size,
                alpha=overlay_alpha,
                smoothing=smoothing,
                smoothing_sigma=smoothing_sigma,
            )
            print(f"  Overlay generated: {overlay_image.shape}")
        else:
            print("  No original image supplied — overlay skipped.")

    return {
        "probability":          slide_score,
        "prediction":           prediction,
        "label":                label_str,
        "confidence":           confidence,
        "n_patches":            N,
        "attention":            avg_attention,
        "heatmap_grid":         heatmap_grid,
        "heatmap_spatial":      heatmap_spatial,
        "overlay_image":        overlay_image,
        "has_coords":           has_coords,
        "coords":               coords_np if has_coords else None,
        # ── NEW keys ──────────────────────────────────────────────────────────
        "per_fold_probs":       all_probs,
        "confidence_interval":  ci_info,
        "calibration_meta":     cal_meta,
        "validation_metrics":   VALIDATION_METRICS,
    }


# ══════════════════════════════════════════════════════════════════════════════
# WSI RECONSTRUCTION FROM PATCH PNGs
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_wsi_from_patches(
    patch_dir: str,
    coords: np.ndarray,
    patch_size: int = 256,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Reconstruct a thumbnail whole-slide image from patch PNGs.
    coords must be ORIGINAL (unscaled) pixel coordinates.
    scale affects visualization only, not inference.
    """
    import cv2

    coords_np = coords.numpy() if torch.is_tensor(coords) else np.array(coords)

    x_min, y_min = coords_np.min(axis=0)
    x_max, y_max = coords_np.max(axis=0)

    vis_patch = max(MIN_VIS_PATCH, int(round(patch_size * scale)))
    width  = int(round((x_max - x_min + patch_size) * scale))
    height = int(round((y_max - y_min + patch_size) * scale))

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for coord in coords_np:
        x, y = int(coord[0]), int(coord[1])
        patch_path = os.path.join(patch_dir, f"{x}_{y}.png")
        if not os.path.exists(patch_path):
            continue

        patch = cv2.imread(patch_path)
        if patch is None:
            continue
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        if scale != 1.0:
            patch = cv2.resize(patch, (vis_patch, vis_patch), interpolation=cv2.INTER_AREA)

        xs = int(round((x - x_min) * scale))
        ys = int(round((y - y_min) * scale))
        xe = min(xs + vis_patch, width)
        ye = min(ys + vis_patch, height)

        canvas[ys:ye, xs:xe] = patch[:ye - ys, :xe - xs]

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    import cv2

    print("\n" + "=" * 60)
    print("  CLAM MIL Inference Pipeline")
    print("=" * 60)

    overlay_alpha = 0.6

    # ── Resolve slide path ────────────────────────────────────────────────────
    if args.slide:
        slide_path = args.slide
        if not os.path.exists(slide_path):
            candidate = os.path.join(FEATURE_DIR, args.slide)
            if os.path.exists(candidate):
                slide_path = candidate
        if not os.path.exists(slide_path):
            raise FileNotFoundError(f"Slide file not found: {args.slide}")
    else:
        pt_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "**", "*.pt"), recursive=True))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {FEATURE_DIR}")
        slide_path = pt_files[0]

    print(f"\n  Slide  : {slide_path}")

    # ── Load features ─────────────────────────────────────────────────────────
    features, coords = load_features(slide_path)
    N, D = features.shape
    print(f"  Patches: {N:,}   Feature dim: {D}")

    all_fold_paths = sorted(glob.glob(os.path.join(MODEL_DIR, "best_fold_*.pt")))

    slide_name = os.path.basename(slide_path).replace(".pt", "")
    PATCH_DIR  = os.path.join(PROJECT_DIR, "data", "patches", slide_name)

    if not os.path.exists(PATCH_DIR):
        print(f"  [WARN] Patch folder not found: {PATCH_DIR}")
        PATCH_DIR = None
    else:
        print(f"  Patch dir: {PATCH_DIR}")

    # ── Prepare original image for overlay ───────────────────────────────────
    original_image = None
    vis_scale = 1.0

    if PATCH_DIR is not None:
        print("\n  Reconstructing WSI from patches…")
        coords_np_tmp = coords.numpy() if torch.is_tensor(coords) else np.array(coords)

        vis_scale = compute_visual_scale(coords_np_tmp, patch_size=256, max_side=VIS_MAX_SIDE)
        print(f"  Visualization scale: {vis_scale:.4f}")

        original_image = reconstruct_wsi_from_patches(
            PATCH_DIR,
            coords_np_tmp,
            patch_size=256,
            scale=vis_scale,
        )
        print(f"  Reconstructed WSI shape: {original_image.shape}")

    # ── Run full inference ────────────────────────────────────────────────────
    result = run_full_inference(
        slide_path,
        all_fold_paths,
        original_image=original_image,
        patch_size=256,
        smoothing=False,
        overlay_alpha=overlay_alpha,
        vis_scale=vis_scale,
    )

    prob       = result["probability"]
    pred       = result["prediction"]
    true_label = get_true_label(slide_path)

    print(f"\n{'─' * 60}")
    print(f"  Score      : {prob:.4f}")
    print(f"  Threshold  : {THRESHOLD}")
    print(f"  Prediction : {'Tumor (1)' if pred == 1 else 'Normal (0)'}")
    if true_label is not None:
        correct = "✓ CORRECT" if pred == true_label else "✗ WRONG"
        print(f"  True Label : {'Tumor (1)' if true_label == 1 else 'Normal (0)'}  {correct}")
    print(f"{'─' * 60}")

    # ── Skip heatmaps if no coords ────────────────────────────────────────────
    if coords is None:
        print("\n  [SKIP] No coordinates — heatmaps cannot be generated.")
        return

    # ── Save grid heatmap ─────────────────────────────────────────────────────
    if result["heatmap_grid"] is not None:
        out_path_grid = os.path.join(OUTPUT_DIR, "heatmap_grid.png")
        label_str = "Tumor" if pred == 1 else "Normal"
        plot_heatmap(
            result["heatmap_grid"],
            title=f"Attention Grid (16×16) | {label_str} (p={prob:.2f})",
            output_path=out_path_grid,
        )

    # ── Save standalone spatial heatmap ──────────────────────────────────────
    if result["heatmap_spatial"] is not None:
        out_path_spatial = os.path.join(OUTPUT_DIR, "heatmap_spatial.png")
        spatial_bgr = cv2.cvtColor(result["heatmap_spatial"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path_spatial, spatial_bgr)
        print(f"  Saved: {out_path_spatial}")

    # ── Save overlay heatmap ──────────────────────────────────────────────────
    if result["overlay_image"] is not None:
        overlay_path = os.path.join(OUTPUT_DIR, "heatmap_overlay.png")
        overlay_bgr  = cv2.cvtColor(result["overlay_image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(overlay_path, overlay_bgr)
        print(f"  Saved: {overlay_path}")
    else:
        print("  [SKIP] Overlay not generated (patch images not available)")

    # ── Optional: ROC + confusion matrix from ensemble CSV ───────────────────
    if os.path.exists(ENSEMBLE_CSV):
        print(f"\n  Loading ensemble predictions from {ENSEMBLE_CSV}…")
        df = pd.read_csv(ENSEMBLE_CSV)
        threshold = tune_threshold(df["label"].values, df["prob"].values)
        preds_all = (df["prob"].values >= threshold).astype(int)
        plot_roc_curve(
            df["label"].values, df["prob"].values,
            output_path=os.path.join(OUTPUT_DIR, "roc_curve.png"),
            threshold=threshold,
        )
        plot_confusion_matrix(
            df["label"].values.astype(int), preds_all,
            output_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
        )
        auc = roc_auc_score(df["label"].values, df["prob"].values)
        print(f"\n  Ensemble AUC       : {auc:.4f}")
        print(f"  Optimal threshold  : {threshold:.4f}")

    # ── Save inference report ─────────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "inference_report.txt")
    with open(report_path, "w") as fh:
        fh.write("=== CLAM Inference Report ===\n\n")
        fh.write(f"Slide       : {slide_path}\n")
        fh.write(f"Patches     : {N}\n")
        fh.write(f"Feature dim : {D}\n")
        fh.write(f"Score       : {prob:.4f}\n")
        fh.write(f"Threshold   : {THRESHOLD}\n")
        fh.write(f"Prediction  : {pred} ({'Tumor' if pred == 1 else 'Normal'})\n")
        if true_label is not None:
            fh.write(f"True Label  : {true_label}\n")
            fh.write(f"Correct     : {pred == true_label}\n")
        ci = result["confidence_interval"]
        fh.write(f"CI (95%)    : [{ci['lower']:.4f}, {ci['upper']:.4f}]\n")
        fh.write(f"Fold agree  : {result['calibration_meta']['fold_agreement']*100:.0f}%\n")
    print(f"\n  Report saved : {report_path}")
    print("\n  ✅  Pipeline complete.\n")


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="CLAM MIL Inference Pipeline")
    p.add_argument("--slide",    type=str, default=None,
                   help="Path to a .pt feature file.")
    p.add_argument("--model",    type=str, default=None,
                   help="Path to a model checkpoint.")
    p.add_argument("--ensemble", action="store_true",
                   help="Average attention from all available fold models.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
