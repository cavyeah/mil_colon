print("Script Started: trainig clam")
import os
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.clam_model import CLAM_SB

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = r"C:\mil_colon\data\features"
LABEL_FILE = os.path.join(ROOT_DIR, "data", "labels.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "src", "models", "five_fold")
os.makedirs(MODEL_DIR, exist_ok=True)

EPOCHS = 17
LR = 2e-5
SEED = 42
MIN_PATCHES = 50
MAX_PATCHES = 5000
N_SPLITS = 5
NUM_WORKERS = 2  # GTX 1650: reduced to avoid CPU/GPU bottleneck
USE_MIXED_PRECISION = True  # Enable for GTX 1650 memory efficiency

# Feature caching mode: "none" (always disk), "lru" (bounded RAM cache), "all" (preload full dataset)
CACHE_MODE = "lru"
MAX_CACHE_ITEMS = 256

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
SPLIT_NAME = "65_15_20"

DEFAULT_THRESHOLD = 0.76
THRESHOLD_CANDIDATES = np.unique(
    np.concatenate(
        [
            np.round(np.arange(0.05, 0.951, 0.01), 2),
            np.array([DEFAULT_THRESHOLD], dtype=np.float32),
        ]
    )
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.cuda.empty_cache()
scaler = torch.amp.GradScaler("cuda") if USE_MIXED_PRECISION else None
METRICS_REPORT_PATH = os.path.join(MODEL_DIR, f"split_{SPLIT_NAME}_metrics_report.txt")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MILDataset(Dataset):
    def __init__(self, feature_dir: str, csv_file: str):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Missing labels file: {csv_file}")

        self.feature_dir = feature_dir
        self.cache = OrderedDict() if CACHE_MODE == "lru" else {}
        df = pd.read_csv(csv_file)

        df = df[["slide_id", "label"]].dropna().copy()
        df["slide_id"] = df["slide_id"].astype(str).str.strip()
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        df["feature_path"] = df["slide_id"].map(
            lambda s: os.path.join(self.feature_dir, f"{s}.pt"),
        )
        df = df[df["feature_path"].map(os.path.exists)].copy()

        self.df = df.reset_index(drop=True)
        self.labels = self.df["label"].tolist()

        if CACHE_MODE == "all":
            print("🚀 Preloading all features into RAM...")
            for path in self.df["feature_path"]:
                self.cache[path] = torch.load(path, map_location="cpu")
            print(f"✅ Done preloading {len(self.cache)} feature files")

    def __len__(self):
        return len(self.df)

    def _load_features(self, path: str):
        if CACHE_MODE == "none":
            return torch.load(path, map_location="cpu")

        if path in self.cache:
            if CACHE_MODE == "lru":
                self.cache.move_to_end(path)
            return self.cache[path]

        feat = torch.load(path, map_location="cpu")
        self.cache[path] = feat

        if CACHE_MODE == "lru" and len(self.cache) > MAX_CACHE_ITEMS:
            self.cache.popitem(last=False)

        return feat

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["feature_path"]

        features = self._load_features(path)

        if isinstance(features, dict):
            features = features.get("features", features.get("feats"))

        features = features.float()

        # STRIDED SAMPLING
        if features.shape[0] > MAX_PATCHES:
            step = max(1, features.shape[0] // MAX_PATCHES)
            idxs = torch.arange(0, features.shape[0], step)[:MAX_PATCHES]
            features = features[idxs]

        return features, torch.tensor(float(row["label"]), dtype=torch.float32)


def compute_auc(labels, probs):
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    pos = labels == 1
    neg = labels == 0

    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan

    ranks = pd.Series(probs).rank().to_numpy()
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


def compute_binary_metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)

    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    accuracy = (tp + tn) / len(labels)
    sensitivity = tp / (tp + fn) if tp + fn > 0 else np.nan
    specificity = tn / (tn + fp) if tn + fp > 0 else np.nan
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan

    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision and sensitivity
        else np.nan
    )

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": compute_auc(labels, probs),
    }


def tune_threshold(labels, probs):
    best = None
    for t in THRESHOLD_CANDIDATES:
        m = compute_binary_metrics(labels, probs, t)
        score = (m["f1"], m["accuracy"])
        if best is None or score > best[0]:
            best = (score, t, m)
    return {"threshold": best[1], **best[2]}


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    probs, labels, attentions = [], [], []

    for i, (feats, label) in enumerate(loader):
        if i % 20 == 0:
            print(f"  Processing batch {i}/{len(loader)}...")
        feats = feats.squeeze(0).to(device)
        label = label.view(1).to(device)

        if feats.shape[0] < MIN_PATCHES:
            continue

        if is_train and USE_MIXED_PRECISION and device.type == "cuda":
            # Mixed precision for training (reduces memory, faster on RTX GPUs)
            with torch.amp.autocast("cuda"):
                logits, attention = model(feats)
                loss = criterion(logits, label)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, attention = model(feats)
            loss = criterion(logits, label)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients
                optimizer.step()

        # Detach to save memory
        probs.append(torch.sigmoid(logits).detach().item())
        labels.append(int(label.item()))
        # Store attention weights for heatmap visualization (normalize to 0-1)
        if attention is not None:
            att_weights = torch.softmax(attention.squeeze(), dim=0).detach().cpu().numpy()
            attentions.append(att_weights)
        
        # Clear GPU cache every 50 batches to prevent memory issues on GTX 1650
        if device.type == "cuda" and (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    return np.array(labels), np.array(probs), attentions


def train_one_fold(dataset, train_idx, val_idx, test_idx, fold_id):
    model = CLAM_SB().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = torch.nn.BCEWithLogitsLoss()

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=1,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=1,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )

    # EARLY STOPPING
    patience, best_loss, counter = 5, float("inf"), 0
    best_model_path = os.path.join(MODEL_DIR, f"fold_{fold_id}_best.pt")

    for epoch in range(EPOCHS):
        run_epoch(model, train_loader, criterion, optimizer)
        val_labels, val_probs, _ = run_epoch(model, val_loader, criterion)

        val_loss = ((val_probs - val_labels) ** 2).mean()
        scheduler.step(val_loss)  # Reduce LR if val_loss plateaus

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1

        if counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}, best loss: {best_loss:.4f}")
            break

    # Load best model for testing
    model.load_state_dict(torch.load(best_model_path))
    return model


def train_five_fold():
    dataset = MILDataset(FEATURE_DIR, LABEL_FILE)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    folds = np.array_split(indices, N_SPLITS)
    all_predictions = []

    for i in range(N_SPLITS):
        print(f"\n{'='*50}")
        print(f"Fold {i+1}/{N_SPLITS}")
        print(f"{'='*50}")
        test_idx = folds[i]
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i])

        split = int(len(train_idx) * 0.85)
        val_idx = train_idx[split:]
        train_idx = train_idx[:split]

        model = train_one_fold(dataset, train_idx, val_idx, test_idx, fold_id=i)

        test_loader = DataLoader(
            Subset(dataset, test_idx),
            batch_size=1,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
            persistent_workers=NUM_WORKERS > 0,
        )
        labels, probs, attentions = run_epoch(model, test_loader, None)

        df = pd.DataFrame({
            "slide_id": dataset.df.iloc[test_idx]["slide_id"].values,
            "label": labels,
            "prob": probs
        })
        all_predictions.append(df)
        
        # Save attention weights per fold for heatmap generation
        attention_file = os.path.join(MODEL_DIR, f"fold_{i}_attentions.npy")
        np.save(attention_file, np.array(attentions, dtype=object), allow_pickle=True)
        print(f"  Saved attention weights: {attention_file}")

    ensemble_df = pd.concat(all_predictions)

    # GLOBAL THRESHOLD
    global_metrics = tune_threshold(
        ensemble_df["label"].values,
        ensemble_df["prob"].values
    )
    threshold = global_metrics["threshold"]

    # ENSEMBLE
    grouped = ensemble_df.groupby("slide_id").mean().reset_index()

    final_metrics = compute_binary_metrics(
        grouped["label"].values,
        grouped["prob"].values,
        threshold
    )

    print("\nFINAL RESULTS")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Sensitivity: {final_metrics['sensitivity']:.4f}")
    print(f"Specificity: {final_metrics['specificity']:.4f}")
    print(f"F1-Score: {final_metrics['f1']:.4f}")
    print(f"AUC: {final_metrics['auc']:.4f}")
    print(f"Optimal Threshold: {threshold:.4f}")
    
    # Save predictions for heatmap generation
    predictions_file = os.path.join(MODEL_DIR, "ensemble_predictions.csv")
    ensemble_df.to_csv(predictions_file, index=False)
    print(f"Ensemble predictions saved: {predictions_file}")
    
    # Save metrics to file
    with open(METRICS_REPORT_PATH, "w") as f:
        f.write("=== 5-FOLD CROSS-VALIDATION RESULTS ===\n\n")
        f.write(f"Accuracy: {final_metrics['accuracy']:.4f}\n")
        f.write(f"Sensitivity: {final_metrics['sensitivity']:.4f}\n")
        f.write(f"Specificity: {final_metrics['specificity']:.4f}\n")
        f.write(f"Precision: {final_metrics['precision']:.4f}\n")
        f.write(f"F1-Score: {final_metrics['f1']:.4f}\n")
        f.write(f"AUC: {final_metrics['auc']:.4f}\n")
        f.write(f"Optimal Threshold: {threshold:.4f}\n")
        f.write(f"\nPredictions file: {predictions_file}\n")
        f.write(f"Attention weights: fold_*_attentions.npy\n")
    print(f"Metrics saved to {METRICS_REPORT_PATH}")


def main():
    print("Inside main")
    print(f"Device: {device}")
    print(f"Mixed Precision: {USE_MIXED_PRECISION}")
    print(f"Num Workers: {NUM_WORKERS}")
    set_seed(SEED)
    try:
        train_five_fold()
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            print("\nGPU cache cleared")


if __name__ == "__main__":
    main()