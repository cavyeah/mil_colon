import os
import openslide
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd

# ======================================================
# CONFIGURATION
# ======================================================

PATCH_SIZE = 256
STRIDE = 256
LEVEL = 0

WSI_ROOT = "data/wsi"  # scans normal + tumor folders recursively
PATCH_DIR = "data/patches/tcga_coad"
LABEL_FILE = "data/labels.csv"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

os.makedirs(PATCH_DIR, exist_ok=True)

# ======================================================
# Tissue Detection
# ======================================================

def is_tissue(patch, white_threshold=0.8):
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    white_pixels = np.sum(gray > 220)
    total_pixels = gray.size
    return (white_pixels / total_pixels) < white_threshold


# ======================================================
# Label Extraction
# ======================================================

def get_label_from_path(slide_path):
    """
    Priority:
      1) TCGA barcode in filename: -01Z- (tumor), -11Z- (normal)
      2) Folder fallback: /tumor|/tumour => 1, /normal => 0
    """
    name = os.path.basename(slide_path).lower()
    parts = [p.lower() for p in os.path.normpath(slide_path).split(os.sep)]

    if "-01z-" in name:
        return 1
    if "-11z-" in name:
        return 0
    if "tumor" in parts or "tumour" in parts:
        return 1
    if "normal" in parts:
        return 0
    return -1


# ======================================================
# WSI Discovery
# ======================================================

def find_svs_files(root_dir):
    svs_paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".svs"):
                svs_paths.append(os.path.join(root, f))
    return sorted(svs_paths, key=lambda p: p.lower())


def is_slide_processed(slide_patch_dir):
    if not os.path.isdir(slide_patch_dir):
        return False
    files = os.listdir(slide_patch_dir)
    has_coords = "coords.npy" in files
    has_images = any(os.path.splitext(f)[1].lower() in IMG_EXTS for f in files)
    return has_coords and has_images


def clear_partial_outputs(slide_patch_dir):
    """
    Clears only patch images + coords.npy if re-processing is needed.
    """
    os.makedirs(slide_patch_dir, exist_ok=True)
    for name in os.listdir(slide_patch_dir):
        path = os.path.join(slide_patch_dir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if name == "coords.npy" or ext in IMG_EXTS:
            os.remove(path)


# ======================================================
# Patch Extraction
# ======================================================

def extract_patches(slide_path):
    slide_name = os.path.basename(slide_path)
    slide_id = os.path.splitext(slide_name)[0]

    print(f"\nProcessing slide: {slide_id}")

    slide_out_dir = os.path.join(PATCH_DIR, slide_id)
    clear_partial_outputs(slide_out_dir)

    slide = openslide.OpenSlide(slide_path)
    try:
        width, height = slide.level_dimensions[LEVEL]

        x_range = range(0, width - PATCH_SIZE + 1, STRIDE)
        y_range = range(0, height - PATCH_SIZE + 1, STRIDE)

        coords = []
        patch_count = 0

        for y in tqdm(y_range, desc=f"{slide_id} rows", leave=False):
            for x in x_range:
                region = slide.read_region((x, y), LEVEL, (PATCH_SIZE, PATCH_SIZE))
                patch = np.array(region)[:, :, :3]

                if not is_tissue(patch):
                    continue

                # coordinate-based filename for spatial traceability
                patch_filename = f"{x}_{y}.png"
                patch_path = os.path.join(slide_out_dir, patch_filename)

                cv2.imwrite(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                coords.append([x, y])
                patch_count += 1

    finally:
        slide.close()

    coords = np.array(coords, dtype=np.int32)
    coord_file = os.path.join(slide_out_dir, "coords.npy")
    np.save(coord_file, coords)

    print(f"Saved {patch_count} patches for {slide_id}")
    print(f"Saved coordinates: {coord_file}")

    return slide_id


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    slides = find_svs_files(WSI_ROOT)

    if len(slides) == 0:
        print("No SVS files found.")
        raise SystemExit(1)

    print(f"\nTotal slides found: {len(slides)}")

    # Load existing labels safely
    if os.path.exists(LABEL_FILE):
        existing_df = pd.read_csv(LABEL_FILE)
        if "slide_id" not in existing_df.columns:
            existing_df["slide_id"] = ""
        if "label" not in existing_df.columns:
            existing_df["label"] = -1
        existing_df = existing_df[["slide_id", "label"]]
    else:
        existing_df = pd.DataFrame(columns=["slide_id", "label"])

    known_slide_ids = set(existing_df["slide_id"].astype(str).tolist())
    new_label_entries = []

    for slide_path in slides:
        slide_id = os.path.splitext(os.path.basename(slide_path))[0]
        slide_patch_dir = os.path.join(PATCH_DIR, slide_id)

        if is_slide_processed(slide_patch_dir):
            print(f"Skipping {slide_id} (already processed)")
        else:
            extract_patches(slide_path)

        if slide_id not in known_slide_ids:
            label = get_label_from_path(slide_path)
            if label != -1:
                new_label_entries.append([slide_id, label])
                known_slide_ids.add(slide_id)

    # Update label file
    if new_label_entries:
        new_df = pd.DataFrame(new_label_entries, columns=["slide_id", "label"])
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=["slide_id"], keep="last")
        final_df.to_csv(LABEL_FILE, index=False)
        print(f"\nAdded {len(new_label_entries)} new label entries.")
    else:
        print("\nNo new label entries to add.")

    print("\nAll remaining slides processed successfully.")