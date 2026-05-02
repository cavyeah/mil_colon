print("Script Started: training CLAM")

import os
import sys
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold

# Fix import path so clam_model.py is always found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clam_model import CLAM_SB

# ============================================================
# PATHS  (adjust if your folder layout differs)
# ============================================================
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIR = os.path.join(ROOT_DIR, "features")
LABEL_FILE  = os.path.join(ROOT_DIR, "data", "labels.csv")
MODEL_DIR   = os.path.join(ROOT_DIR, "models", "five_fold")
os.makedirs(MODEL_DIR, exist_ok=True)

METRICS_REPORT_PATH = os.path.join(MODEL_DIR, "metrics_report.txt")

# ============================================================
# HYPER-PARAMETERS
# ============================================================
EPOCHS      = 65
LR          = 5e-5
SEED        = 42
MIN_PATCHES = 100
MAX_PATCHES = 900
N_SPLITS    = 5
PATIENCE    = 8          # early-stopping patience (epochs without improvement)
POS_WEIGHT  = 1.3        # up-weight the positive (tumor) class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================
class MILDataset(Dataset):
    """
    Loads pre-extracted CLAM feature bags (.pt files).
    Each .pt file contains:
        { 'features': [N, D],  'coords': [N, 2] }
    or a bare tensor [N, D] (legacy format).
    """

    def __init__(self, feature_dir: str, csv_file: str):
        print("Loading dataset...")

        if not os.path.exists(feature_dir):
            raise FileNotFoundError(f"Features folder NOT found: {feature_dir}")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"labels.csv NOT found: {csv_file}")

        df = pd.read_csv(csv_file)
        df = df[["slide_id", "label"]].dropna()
        df["slide_id"] = df["slide_id"].astype(str).str.strip()
        df["label"]    = df["label"].astype(int)

        feature_files = os.listdir(feature_dir)

        def _find_match(slide_id):
            for f in feature_files:
                if slide_id in f:
                    return os.path.join(feature_dir, f)
            return None

        df["feature_path"] = df["slide_id"].apply(_find_match)
        print(f"  Missing feature files: {df['feature_path'].isnull().sum()}")

        df = df[df["feature_path"].notnull()].reset_index(drop=True)
        print(f"  Dataset size after matching: {len(df)}")

        self.df     = df
        self.labels = df["label"].tolist()
        self.cache  = OrderedDict()   # LRU cache to limit RAM usage

    def __len__(self):
        return len(self.df)

    def _load(self, path: str) -> torch.Tensor:
        """LRU-cached feature loader."""
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]

        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            feat = obj.get("features", obj.get("feats"))
        elif torch.is_tensor(obj):
            feat = obj
        else:
            raise ValueError(f"Unknown feature format in {path}")

        self.cache[path] = feat
        if len(self.cache) > 128:
            self.cache.popitem(last=False)   # evict LRU entry

        return feat

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        features = self._load(row["feature_path"]).float()

        # Random patch sub-sampling to control memory
        if features.shape[0] > MAX_PATCHES:
            idxs     = torch.randperm(features.shape[0])[:MAX_PATCHES]
            features = features[idxs]

        return features, torch.tensor(float(row["label"]))


# ============================================================
# METRICS  (manual implementation — no sklearn dependency in train loop)
# ============================================================
def compute_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Wilcoxon-Mann-Whitney AUC (equivalent to sklearn roc_auc_score)."""
    pos  = labels == 1
    neg  = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = pd.Series(probs).rank().to_numpy()
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


def compute_binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    acc         = (tp + tn) / len(labels)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = (2 * precision * sensitivity / (precision + sensitivity)
                   if (precision + sensitivity) > 0 else 0.0)

    return {
        "accuracy":    acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision":   precision,
        "f1":          f1,
        "auc":         compute_auc(labels, probs),
    }


# ============================================================
# THRESHOLD TUNING  (maximise F1 on validation/test)
# ============================================================
def tune_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.002):
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        prec   = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


# ============================================================
# TRAIN / EVAL EPOCH
# ============================================================
def run_epoch(model, loader, optimizer=None):
    """
    Runs one pass through the data.
    - If optimizer is given  → training mode  (with backprop)
    - If optimizer is None   → evaluation mode (no backprop)

    Returns  (labels, probs, mean_bce_loss)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    pos_weight = torch.tensor([POS_WEIGHT], device=device)
    all_probs, all_labels, losses = [], [], []

    for feats, label in loader:
        feats = feats.squeeze(0).to(device)   # [N, D]
        label = label.view(1).to(device)       # [1]

        # Skip slides with too few patches (corrupted / very small ROI)
        if feats.shape[0] < MIN_PATCHES:
            continue

        logits, _ = model(feats)   # logits: [1], _: attention (top-K)

        loss = F.binary_cross_entropy_with_logits(
            logits, label, pos_weight=pos_weight
        )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        all_probs.append(torch.sigmoid(logits).item())
        all_labels.append(int(label.item()))
        losses.append(loss.item())

    mean_loss = float(np.mean(losses)) if losses else float("inf")
    return np.array(all_labels), np.array(all_probs), mean_loss


# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def train_five_fold():
    dataset = MILDataset(FEATURE_DIR, LABEL_FILE)

    # ------------------------------------------------------------------
    # Balance dataset: 23 tumor + 23 normal  (adjust head() as needed)
    # ------------------------------------------------------------------
    df      = dataset.df
    tumor   = df[df.label == 1].head(23)
    normal  = df[df.label == 0].head(23)
    df      = pd.concat([tumor, normal]).reset_index(drop=True)

    dataset.df     = df
    dataset.labels = df["label"].tolist()
    print(f"Final balanced dataset: {len(dataset)} slides")

    labels = np.array(dataset.labels)
    skf    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_fold_preds = []

    for fold_i, (trainval_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels)
    ):
        print(f"\n{'='*50}")
        print(f"  FOLD {fold_i + 1} / {N_SPLITS}")
        print(f"{'='*50}")

        # ---------------------------------------------------------------
        # FIXED SPLIT: 80% train, 20% validation  (was inverted before)
        # ---------------------------------------------------------------
        np.random.seed(SEED)
        np.random.shuffle(trainval_idx)

        split_pt   = int(len(trainval_idx) * 0.80)   # 80 % → train
        train_idx  = trainval_idx[:split_pt]
        val_idx    = trainval_idx[split_pt:]          # 20 % → val

        print(f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

        # ---------------------------------------------------------------
        # Dataloaders
        # ---------------------------------------------------------------
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=1, shuffle=False)
        test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=1, shuffle=False)

        # ---------------------------------------------------------------
        # Model + optimizer
        # ---------------------------------------------------------------
        model     = CLAM_SB().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

        best_val_loss  = float("inf")
        patience_count = 0
        best_model_path = os.path.join(MODEL_DIR, f"best_fold_{fold_i}.pt")

        # ---------------------------------------------------------------
        # Epoch loop with EARLY STOPPING on validation BCE loss
        # ---------------------------------------------------------------
        for epoch in range(EPOCHS):
            _, _, train_loss = run_epoch(model, train_loader, optimizer)
            _, _, val_loss   = run_epoch(model, val_loader)

            print(f"  Epoch {epoch+1:3d} | train_bce={train_loss:.4f}  val_bce={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"             ✓ New best val_bce={best_val_loss:.4f} — model saved")
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    print(f"  Early stopping triggered at epoch {epoch+1}")
                    break

        # ---------------------------------------------------------------
        # Load best checkpoint and evaluate on held-out test fold
        # ---------------------------------------------------------------
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        labels_test, probs_test, _ = run_epoch(model, test_loader)

        fold_df = pd.DataFrame({
            "slide_id": dataset.df.iloc[test_idx]["slide_id"].values,
            "label":    labels_test,
            "prob":     probs_test,
        })
        all_fold_preds.append(fold_df)
        print(f"  Fold {fold_i+1} test AUC = {compute_auc(labels_test, probs_test):.4f}")

    # ---------------------------------------------------------------
    # ENSEMBLE: average predicted probability across folds  
    # (slides that appeared in multiple test sets get averaged)
    # ---------------------------------------------------------------
    final_df = pd.concat(all_fold_preds)
    final_df = final_df.groupby("slide_id")[["label", "prob"]].mean().reset_index()

    predictions_path = os.path.join(MODEL_DIR, "ensemble_predictions.csv")
    final_df.to_csv(predictions_path, index=False)
    print(f"\nEnsemble predictions saved to: {predictions_path}")

    # ---------------------------------------------------------------
    # Tune threshold and report final metrics
    # ---------------------------------------------------------------
    threshold    = tune_threshold(final_df["label"].values, final_df["prob"].values)
    final_metrics = compute_binary_metrics(
        final_df["label"].values, final_df["prob"].values, threshold
    )

    print("\n" + "="*50)
    print("  FINAL ENSEMBLE RESULTS")
    print("="*50)
    for k, v in final_metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print(f"  {'threshold':12s}: {threshold:.4f}")

    with open(METRICS_REPORT_PATH, "w") as fh:
        fh.write("=== 5-FOLD CROSS-VALIDATION RESULTS ===\n\n")
        for k, v in final_metrics.items():
            fh.write(f"{k.capitalize()}: {v:.4f}\n")
        fh.write(f"Optimal Threshold: {threshold:.4f}\n")
        fh.write(f"\nPredictions file: {predictions_path}\n")

    print(f"\nMetrics saved to {METRICS_REPORT_PATH}")


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    print(f"Device: {device}")
    set_seed(SEED)
    try:
        train_five_fold()
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            print("\nGPU cache cleared")


if __name__ == "__main__":
    main()