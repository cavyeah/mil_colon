import os
import re
import cv2
import numpy as np
import torch

from clam_model import CLAM_SB

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SLIDE_ID = "TCGA-3L-AA1B-01Z-00-DX2.17CE3683-F4B1-4978-A281-8F620C4D77B4"

FEATURE_DIR = os.path.join(ROOT_DIR, "data", "features")
PATCH_DIR = os.path.join(ROOT_DIR, "data", "patches", "tcga_coad")
MODEL_DIR = os.path.join(ROOT_DIR, "src", "models", "five_fold")
OUTPUT_DIR = os.path.join(ROOT_DIR, "src", "test_heatmap")

MAX_PATCHES = 64
OVERLAY_ALPHA = 0.6
COLORMAP = cv2.COLORMAP_JET

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    models = []
    for fold in range(1, 6):
        path = os.path.join(MODEL_DIR, f"best_clam_fold_{fold}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model: {path}")

        model = CLAM_SB().to(device)
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)

        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        models.append(model)

    print(f"Loaded {len(models)} ensemble models")
    return models


def load_features(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing feature file: {path}")

    features = torch.load(path, map_location="cpu")

    if isinstance(features, dict):
        features = features.get("features", features.get("feats"))

    if features is None or not torch.is_tensor(features):
        raise ValueError(f"Invalid feature tensor in file: {path}")

    features = features.float()
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if features.ndim == 3 and features.shape[0] == 1:
        features = features.squeeze(0)

    if features.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {features.shape}")

    return features


def compute_attention(features, models):
    """
    Uses full attention map from attention head (N values),
    and ensemble probability from standard model forward pass.
    """
    features = features.to(device)

    all_attentions = []
    probs = []

    with torch.inference_mode():
        for model in models:
            logits, _ = model(features)
            probs.append(float(torch.sigmoid(logits).item()))

            # Full attention over all N patches (before top-k truncation)
            raw_A = model.attention_net.attention(features)   # [N, 1]
            raw_A = raw_A - torch.max(raw_A)
            full_A = torch.softmax(raw_A, dim=0).squeeze(1)   # [N]
            all_attentions.append(full_A.detach().cpu().numpy())

    ensemble_attention = np.mean(np.stack(all_attentions, axis=0), axis=0)
    ensemble_attention = (
        (ensemble_attention - ensemble_attention.min())
        / (ensemble_attention.max() - ensemble_attention.min() + 1e-8)
    )

    ensemble_prob = float(np.mean(probs))
    return ensemble_attention, ensemble_prob


def colorise_patch(img, score):
    value = int(float(score) * 255)
    heat = cv2.applyColorMap(np.full(img.shape[:2], value, dtype=np.uint8), COLORMAP)
    return cv2.addWeighted(img, OVERLAY_ALPHA, heat, 1 - OVERLAY_ALPHA, 0)


def natural_sort_key(name):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


def load_patch_files(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Missing patch folder: {folder}")

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ]
    files = sorted(files, key=natural_sort_key)
    return files


def build_grid(patches):
    if len(patches) == 0:
        raise ValueError("No valid patches to render.")

    n = len(patches)
    cols = max(1, int(np.sqrt(n) * 1.5))
    rows = int(np.ceil(n / cols))

    h, w = patches[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for i, patch in enumerate(patches):
        r = i // cols
        c = i % cols
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = patch

    return canvas


def main():
    feature_path = os.path.join(FEATURE_DIR, SLIDE_ID + ".pt")
    patch_folder = os.path.join(PATCH_DIR, SLIDE_ID)

    models = load_models()
    features = load_features(feature_path)
    patch_files = load_patch_files(patch_folder)

    n = min(len(patch_files), features.shape[0], MAX_PATCHES)
    if n == 0:
        raise ValueError("No overlap between feature patches and image patches.")

    print("Feature patches:", features.shape[0])
    print("Image patches:", len(patch_files))
    print("Using:", n)

    features = features[:n]
    patch_files = patch_files[:n]

    attention, prob = compute_attention(features, models)
    attention = attention[:n]

    patches = []
    for patch_name, score in zip(patch_files, attention):
        img = cv2.imread(os.path.join(patch_folder, patch_name))
        if img is None:
            continue
        patches.append(colorise_patch(img, score))

    canvas = build_grid(patches)

    save_path = os.path.join(OUTPUT_DIR, SLIDE_ID + "_heatmap.png")
    cv2.imwrite(save_path, canvas)

    print("\nSlide:", SLIDE_ID)
    print("Patches used:", len(patches))
    print("Tumor probability:", prob)
    print("Heatmap saved:", save_path)


if __name__ == "__main__":
    main()