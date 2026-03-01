import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
PATCH_ROOT = "data/patches/tcga_coad"
FEATURE_ROOT = "data/features"
BATCH_SIZE = 128   # Increase for Lightning GPU
NUM_WORKERS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(FEATURE_ROOT, exist_ok=True)

# ==============================
# MODEL
# ==============================
model = models.resnet50(weights="IMAGENET1K_V2")
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()

# ==============================
# TRANSFORMS
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# DATASET
# ==============================
class PatchDataset(Dataset):
    def __init__(self, patch_paths):
        self.patch_paths = patch_paths

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        img = Image.open(self.patch_paths[idx]).convert("RGB")
        img = transform(img)
        return img

# ==============================
# FEATURE EXTRACTION
# ==============================
@torch.no_grad()
def extract_slide(slide_folder):

    slide_name = os.path.basename(slide_folder)
    save_path = os.path.join(FEATURE_ROOT, slide_name + ".pt")

    if os.path.exists(save_path):
        print(f"[SKIP] {slide_name}")
        return

    patch_paths = [
        os.path.join(slide_folder, f)
        for f in os.listdir(slide_folder)
        if f.endswith(".png") or f.endswith(".jpg")
    ]

    dataset = PatchDataset(patch_paths)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    features = []

    print(f"\nProcessing: {slide_name} ({len(dataset)} patches)")

    for batch in tqdm(loader):
        batch = batch.to(device, non_blocking=True)
        feat = model(batch)
        feat = feat.view(batch.size(0), -1)
        features.append(feat.cpu())

    features = torch.cat(features)
    torch.save(features, save_path)

    print(f"Saved {slide_name} → {features.shape}")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    slide_folders = [
        os.path.join(PATCH_ROOT, d)
        for d in os.listdir(PATCH_ROOT)
        if os.path.isdir(os.path.join(PATCH_ROOT, d))
    ]

    print(f"Slides found: {len(slide_folders)}")

    for slide in slide_folders:
        extract_slide(slide)

    print("\nFeature extraction complete.")