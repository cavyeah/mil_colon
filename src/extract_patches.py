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

SVS_DIR = "data/wsi"
PATCH_DIR = "data/patches/tcga_coad"
LABEL_FILE = "data/labels.csv"

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

def get_label_from_filename(filename):
    if "-01Z-" in filename:
        return 1  # tumor
    elif "-11Z-" in filename:
        return 0  # normal
    else:
        return -1


# ======================================================
# Patch Extraction
# ======================================================

def extract_patches(slide_path):

    slide_name = os.path.basename(slide_path)
    slide_id = slide_name.replace(".svs", "")

    print(f"\nProcessing slide: {slide_id}")

    slide = openslide.OpenSlide(slide_path)
    width, height = slide.level_dimensions[LEVEL]

    slide_out_dir = os.path.join(PATCH_DIR, slide_id)
    os.makedirs(slide_out_dir, exist_ok=True)

    patch_count = 0

    for y in tqdm(range(0, height, STRIDE)):
        for x in range(0, width, STRIDE):

            if x + PATCH_SIZE > width or y + PATCH_SIZE > height:
                continue

            region = slide.read_region(
                (x, y),
                LEVEL,
                (PATCH_SIZE, PATCH_SIZE)
            )

            patch = np.array(region)[:, :, :3]

            if not is_tissue(patch):
                continue

            patch_filename = f"{slide_id}_{patch_count}.png"
            patch_path = os.path.join(slide_out_dir, patch_filename)

            cv2.imwrite(
                patch_path,
                cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            )

            patch_count += 1

    slide.close()

    print(f"Saved {patch_count} patches for {slide_id}")
    return slide_id


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    slides = [f for f in os.listdir(SVS_DIR) if f.endswith(".svs")]

    if len(slides) == 0:
        print("No SVS files found.")
        exit()

    print(f"\nTotal slides found: {len(slides)}")

    # Load existing labels if available
    if os.path.exists(LABEL_FILE):
        existing_df = pd.read_csv(LABEL_FILE)
        processed_slides = set(existing_df["slide_id"].tolist())
    else:
        existing_df = pd.DataFrame(columns=["slide_id", "label"])
        processed_slides = set()

    new_label_entries = []

    for slide_file in slides:

        slide_id = slide_file.replace(".svs", "")
        slide_patch_dir = os.path.join(PATCH_DIR, slide_id)

        # Skip if patches already exist
        if os.path.exists(slide_patch_dir) and len(os.listdir(slide_patch_dir)) > 0:
            print(f"Skipping {slide_id} (already processed)")
            continue

        slide_path = os.path.join(SVS_DIR, slide_file)

        extract_patches(slide_path)

        label = get_label_from_filename(slide_file)
        if label != -1:
            new_label_entries.append([slide_id, label])

    # Append new labels safely
    if new_label_entries:
        new_df = pd.DataFrame(new_label_entries, columns=["slide_id", "label"])
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df.to_csv(LABEL_FILE, index=False)

    print("\nAll remaining slides processed successfully.")
