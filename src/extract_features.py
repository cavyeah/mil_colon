import os
import re
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ==============================
6


# CONFIG
# ==============================
PATCH_ROOT   = "data/patches/tcga_coad"
FEATURE_ROOT = "data/features"
BATCH_SIZE   = 64
NUM_WORKERS  = 4
IMG_EXTS     = {".png", ".jpg", ".jpeg"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
os.makedirs(FEATURE_ROOT, exist_ok=True)

# ==============================
# MODEL
# ==============================
_base  = models.resnet50(weights="IMAGENET1K_V2")
model  = nn.Sequential(*list(_base.children())[:-1]).to(device)
model.eval()

# ==============================
# TRANSFORMS
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ==============================
# NATURAL SORT
# ==============================
def natural_sort_key(name: str):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", name)]


COORD_PATTERN = re.compile(r"(-?\d+)[_x,](-?\d+)$")


def parse_coords_from_filenames(patch_files: list):
    parsed = []
    for fname in patch_files:
        stem = os.path.splitext(fname)[0]
        m = COORD_PATTERN.search(stem)
        if m is None:
            return None
        parsed.append((int(m.group(1)), int(m.group(2))))
    return np.asarray(parsed, dtype=np.int64)


def sorted_coords_rows(coords: np.ndarray):
    order = np.lexsort((coords[:, 0], coords[:, 1]))
    return coords[order]


def sort_patch_files_with_coords(patch_files: list):
    """
    Prefer row-major sorting by filename coordinates (y, x) so ordering matches
    extract_patches.py coordinate generation.
    """
    filename_coords = parse_coords_from_filenames(patch_files)
    if filename_coords is None:
        return sorted(patch_files, key=natural_sort_key), None

    order = np.lexsort((filename_coords[:, 0], filename_coords[:, 1]))
    sorted_files = [patch_files[i] for i in order]
    return sorted_files, filename_coords[order]


# ==============================
# COORDINATE LOADING
# ==============================

def load_coords_for_slide(slide_folder: str, patch_files: list, filename_coords=None):
    """
    Returns np.ndarray (N, 2) of (x, y) pixel coords aligned to patch_files order.

    Priority:
      1. coords.npy  saved by extract_patches.py
      2. Coordinates encoded in filenames  e.g. slideID_x_y.png
      3. Raise — do not silently fall back to a wrong ordering
    """
    coord_path = os.path.join(slide_folder, "coords.npy")

    # --- Option 1: coords.npy -------------------------------------------------
    if os.path.exists(coord_path):
        coords = np.load(coord_path)
        if coords.ndim == 2 and coords.shape[1] >= 2 and len(coords) == len(patch_files):
            coords = coords[:, :2].astype(np.int64)

            if filename_coords is None:
                return coords

            if np.array_equal(coords, filename_coords):
                return coords

            if np.array_equal(sorted_coords_rows(coords), sorted_coords_rows(filename_coords)):
                print("  [INFO] coords.npy order differs from patch file order; "
                      "using filename-derived coordinate order for alignment.")
                return filename_coords

            print(f"  [WARN] coords.npy values differ from filename-derived coordinates "
                  f"for {os.path.basename(slide_folder)}; "
                  f"trusting filename-derived coordinates.")
            return filename_coords

        print(f"  [WARN] coords.npy shape {coords.shape} "
              f"does not match {len(patch_files)} patches — trying filename parse.")

    # --- Option 2: parse from filenames  e.g. *_1024_2048.png ----------------
    if filename_coords is not None:
        return filename_coords

    parsed = parse_coords_from_filenames(patch_files)
    if parsed is not None:
        return parsed

    # --- Fail clearly ---------------------------------------------------------
    raise RuntimeError(
        f"Cannot resolve spatial coordinates for {slide_folder}.\n"
        f"  coords.npy missing or misaligned, "
        f"and only 0/{len(patch_files)} filenames carry coordinates.\n"
        f"  Re-run extract_patches.py which saves coords.npy automatically."
    )


# ==============================
# DATASET
# ==============================
class PatchDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return transform(img), idx
        except Exception:
            return torch.zeros(3, 224, 224), idx


# ==============================
# FEATURE EXTRACTION
# ==============================
@torch.no_grad()
def extract_slide(slide_folder: str):
    slide_name = os.path.basename(slide_folder)
    save_path  = os.path.join(FEATURE_ROOT, slide_name + ".pt")

    if os.path.exists(save_path):
        saved = None
        try:
            saved = torch.load(save_path, map_location="cpu")
        except Exception as e:
            print(f"[WARN] {slide_name} existing feature file unreadable ({e}); re-extracting")

        # already new format
        if isinstance(saved, dict) and "features" in saved and "coords" in saved:
            print(f"[SKIP] {slide_name} (already has features + coords)")
            return
        print(f"[UPGRADE] {slide_name} — re-extracting to add coords")

    # ---- collect patch files and keep coordinate-safe ordering ----------------
    patch_files, filename_coords = sort_patch_files_with_coords(
        [f for f in os.listdir(slide_folder)
         if os.path.splitext(f)[1].lower() in IMG_EXTS],
    )

    if len(patch_files) == 0:
        print(f"[SKIP] {slide_name} — no patch images found")
        return

    # ---- load coordinates BEFORE extraction so we fail fast ------------------
    coords = load_coords_for_slide(
        slide_folder,
        patch_files,
        filename_coords=filename_coords,
    )   # (N, 2)

    patch_paths = [os.path.join(slide_folder, f) for f in patch_files]

    dataset = PatchDataset(patch_paths)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,          # must stay False to preserve spatial order
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    all_features = []
    all_indices  = []

    print(f"\nProcessing: {slide_name}  ({len(dataset)} patches)")

    for batch_imgs, batch_idx in tqdm(loader):
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        feat = model(batch_imgs).view(batch_imgs.size(0), -1)

        all_features.append(feat.cpu())
        all_indices.extend(batch_idx.tolist())

    all_features = torch.cat(all_features, dim=0)   # (N, 2048)

    # reorder by original index (DataLoader workers can return out of order)
    order        = sorted(range(len(all_indices)), key=lambda i: all_indices[i])
    all_features = all_features[order]
    coords       = coords[order]

    assert len(all_features) == len(coords), (
        f"Feature/coord mismatch after reorder: "
        f"{len(all_features)} vs {len(coords)}"
    )

    # ---- save as dict with features + coords ---------------------------------
    payload = {
        "features": all_features,          # torch.Tensor  (N, 2048)
        "coords":   torch.from_numpy(coords),  # torch.Tensor  (N, 2)  int64
    }
    torch.save(payload, save_path)

    print(f"Saved {slide_name}  features={tuple(all_features.shape)}  "
          f"coords={tuple(coords.shape)}")

    del all_features, coords, payload
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    slide_folders = sorted(
        [os.path.join(PATCH_ROOT, d) for d in os.listdir(PATCH_ROOT)
         if os.path.isdir(os.path.join(PATCH_ROOT, d))]
    )

    print(f"Slides found: {len(slide_folders)}")

    for slide in slide_folders:
        extract_slide(slide)

    print("\nFeature extraction complete.")