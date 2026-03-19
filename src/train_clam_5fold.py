import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.clam_model import CLAM_SB

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(ROOT_DIR, "data", "features")
LABEL_FILE = os.path.join(ROOT_DIR, "data", "labels.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "src", "models", "five_fold")
os.makedirs(MODEL_DIR, exist_ok=True)

EPOCHS = 30
LR = 1e-5
SEED = 42
MIN_PATCHES = 10
N_SPLITS = 5

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
SPLIT_NAME = "70_10_20"

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
        df = pd.read_csv(csv_file)

        required = {"slide_id", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"labels.csv is missing required columns: {sorted(missing)}")

        df = df[["slide_id", "label"]].dropna().copy()
        df["slide_id"] = df["slide_id"].astype(str).str.strip()
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        unique_labels = set(df["label"].unique().tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Only binary labels 0/1 are supported. Found: {sorted(unique_labels)}")

        df["feature_path"] = df["slide_id"].map(
            lambda s: os.path.join(self.feature_dir, f"{s}.pt"),
        )
        exists_mask = df["feature_path"].map(os.path.exists)
        missing_count = int((~exists_mask).sum())
        if missing_count > 0:
            print(f"[WARN] Skipping {missing_count} slides with missing feature files.")
            df = df[exists_mask].copy()

        self.df = df.reset_index(drop=True)
        self.slides = self.df["slide_id"].tolist()
        self.labels = self.df["label"].tolist()

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _to_feature_tensor(obj):
        if isinstance(obj, dict):
            features = obj.get("features", obj.get("feats"))
        else:
            features = obj

        if not torch.is_tensor(features):
            raise TypeError("Feature payload does not contain a tensor.")

        features = features.float()
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)

        if features.ndim != 2:
            raise ValueError(f"Expected 2D feature tensor, got shape {tuple(features.shape)}")

        if features.shape[0] == 0:
            raise ValueError("Feature tensor is empty.")

        return features

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_path = row["feature_path"]
        label = float(row["label"])

        loaded = torch.load(feature_path, map_location="cpu")
        features = self._to_feature_tensor(loaded)

        return features, torch.tensor(label, dtype=torch.float32)


def make_stratified_folds(labels, n_splits=5, seed=42):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    folds = [[] for _ in range(n_splits)]

    for class_label in np.unique(labels):
        class_indices = np.where(labels == class_label)[0]
        rng.shuffle(class_indices)

        for fold_idx, split in enumerate(np.array_split(class_indices, n_splits)):
            folds[fold_idx].extend(split.tolist())

    return [np.array(sorted(fold), dtype=int) for fold in folds]


def make_stratified_train_val_split(indices, labels, val_ratio=0.125, seed=42):
    indices = np.asarray(indices, dtype=int)
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)

    train_idx = []
    val_idx = []

    for class_label in np.unique(labels):
        class_mask = labels == class_label
        class_indices = indices[class_mask]
        rng.shuffle(class_indices)

        n_class = len(class_indices)
        n_val = int(np.floor(n_class * val_ratio))

        if n_class >= 2:
            n_val = max(1, min(n_class - 1, n_val))
        else:
            n_val = 0

        val_idx.extend(class_indices[:n_val].tolist())
        train_idx.extend(class_indices[n_val:].tolist())

    train_idx = np.array(sorted(train_idx), dtype=int)
    val_idx = np.array(sorted(val_idx), dtype=int)

    # Fallback for tiny cohorts where stratification may still produce empty val.
    if len(val_idx) == 0 and len(train_idx) > 1:
        val_idx = np.array([train_idx[-1]], dtype=int)
        train_idx = train_idx[:-1]

    return train_idx, val_idx


def validate_fold_split(train_idx, val_idx, test_idx, n_total):
    covered = np.concatenate([train_idx, val_idx, test_idx])
    if len(covered) != n_total:
        raise RuntimeError("Fold split does not cover all samples.")

    if len(np.unique(covered)) != n_total:
        raise RuntimeError("Fold split contains duplicate sample assignments.")

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(
            "One of train/val/test splits is empty. Increase dataset size or adjust ratios.",
        )


def split_label_counts(labels, indices):
    subset = np.asarray(labels, dtype=np.int64)[np.asarray(indices, dtype=int)]
    n_pos = int(np.sum(subset == 1))
    n_neg = int(np.sum(subset == 0))
    return len(indices), n_neg, n_pos


def build_criterion(train_labels):
    train_labels = np.asarray(train_labels)
    pos = int(np.sum(train_labels == 1))
    neg = int(np.sum(train_labels == 0))

    if pos == 0 or neg == 0:
        return torch.nn.BCEWithLogitsLoss()

    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def compute_auc(labels, probs):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)

    if labels.size == 0:
        return np.nan

    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = pd.Series(probs).rank(method="average").to_numpy()
    pos_rank_sum = ranks[pos_mask].sum()
    auc = (pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)

    return float(auc)


def empty_metric_bundle(threshold):
    return {
        "threshold": float(threshold),
        "accuracy": np.nan,
        "sensitivity": np.nan,
        "specificity": np.nan,
        "precision": np.nan,
        "f1": np.nan,
        "youden_j": np.nan,
        "auc": np.nan,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }


def compute_binary_metrics(labels, probs, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)

    if labels.size == 0:
        return empty_metric_bundle(threshold)

    preds = (probs >= threshold).astype(np.int64)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    total = labels.size
    accuracy = (tp + tn) / total if total > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    if pd.isna(precision) or pd.isna(sensitivity) or (precision + sensitivity) == 0:
        f1 = np.nan
    else:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)

    if pd.isna(sensitivity) or pd.isna(specificity):
        youden_j = np.nan
    else:
        youden_j = sensitivity + specificity - 1

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy) if not pd.isna(accuracy) else np.nan,
        "sensitivity": float(sensitivity) if not pd.isna(sensitivity) else np.nan,
        "specificity": float(specificity) if not pd.isna(specificity) else np.nan,
        "precision": float(precision) if not pd.isna(precision) else np.nan,
        "f1": float(f1) if not pd.isna(f1) else np.nan,
        "youden_j": float(youden_j) if not pd.isna(youden_j) else np.nan,
        "auc": compute_auc(labels, probs),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def tune_threshold(labels, probs, preferred_threshold=DEFAULT_THRESHOLD):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)

    if labels.size == 0:
        return empty_metric_bundle(preferred_threshold)

    best_metrics = None
    best_score = None

    for threshold in THRESHOLD_CANDIDATES:
        metrics = compute_binary_metrics(labels, probs, float(threshold))

        # Primary: Youden's J
        # Tie-breakers: F1, accuracy, closeness to 0.76
        score = (
            metrics["youden_j"] if not pd.isna(metrics["youden_j"]) else -np.inf,
            metrics["f1"] if not pd.isna(metrics["f1"]) else -np.inf,
            metrics["accuracy"] if not pd.isna(metrics["accuracy"]) else -np.inf,
            -abs(float(threshold) - float(preferred_threshold)),
        )

        if best_score is None or score > best_score:
            best_score = score
            best_metrics = metrics

    return best_metrics


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    decision_threshold=DEFAULT_THRESHOLD,
    sample_indices=None,
):
    is_train = optimizer is not None

    if is_train:
        model.train()
        context = torch.enable_grad()
    else:
        model.eval()
        context = torch.no_grad()

    total_loss = 0.0
    processed_batches = 0
    skipped_batches = 0

    all_probs = []
    all_labels = []
    used_indices = []

    with context:
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 3:
                feats, label, _ = batch
            else:
                feats, label = batch

            feats = feats.squeeze(0).to(device)
            label = label.view(1).to(device)

            if feats.shape[0] < MIN_PATCHES:
                skipped_batches += 1
                continue

            if is_train:
                optimizer.zero_grad()

            logits, _ = model(feats)
            loss = criterion(logits, label)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                skipped_batches += 1
                continue

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            prob = float(torch.sigmoid(logits).view(-1).detach().cpu().item())
            target = int(label.view(-1).detach().cpu().item())

            all_probs.append(prob)
            all_labels.append(target)

            if sample_indices is not None:
                used_indices.append(int(sample_indices[batch_idx]))

            total_loss += float(loss.item())
            processed_batches += 1

    if processed_batches == 0:
        base_metrics = empty_metric_bundle(decision_threshold)
        return {
            "loss": np.nan,
            "threshold": base_metrics["threshold"],
            "accuracy": base_metrics["accuracy"],
            "sensitivity": base_metrics["sensitivity"],
            "specificity": base_metrics["specificity"],
            "precision": base_metrics["precision"],
            "f1": base_metrics["f1"],
            "youden_j": base_metrics["youden_j"],
            "auc": base_metrics["auc"],
            "processed_batches": 0,
            "skipped_batches": skipped_batches,
            "labels": np.asarray([], dtype=np.int64),
            "probs": np.asarray([], dtype=np.float32),
            "indices": np.asarray([], dtype=np.int64),
        }

    base_metrics = compute_binary_metrics(all_labels, all_probs, decision_threshold)

    return {
        "loss": total_loss / processed_batches,
        "threshold": base_metrics["threshold"],
        "accuracy": base_metrics["accuracy"],
        "sensitivity": base_metrics["sensitivity"],
        "specificity": base_metrics["specificity"],
        "precision": base_metrics["precision"],
        "f1": base_metrics["f1"],
        "youden_j": base_metrics["youden_j"],
        "auc": base_metrics["auc"],
        "processed_batches": processed_batches,
        "skipped_batches": skipped_batches,
        "labels": np.asarray(all_labels, dtype=np.int64),
        "probs": np.asarray(all_probs, dtype=np.float32),
        "indices": np.asarray(used_indices, dtype=np.int64),
    }


def build_prediction_frame(
    dataset,
    indices,
    labels,
    probs,
    split_name,
    tuned_threshold,
    fold_number=None,
):
    if len(labels) == 0:
        return pd.DataFrame(
            columns=[
                "fold",
                "split",
                "slide_id",
                "label",
                "prob",
                "default_threshold",
                "pred_at_default",
                "tuned_threshold",
                "pred_at_tuned",
            ]
        )

    slide_ids = dataset.df.iloc[indices]["slide_id"].tolist()
    probs = np.asarray(probs, dtype=np.float32)

    return pd.DataFrame(
        {
            "fold": fold_number,
            "split": split_name,
            "slide_id": slide_ids,
            "label": np.asarray(labels, dtype=np.int64),
            "prob": probs,
            "default_threshold": DEFAULT_THRESHOLD,
            "pred_at_default": (probs >= DEFAULT_THRESHOLD).astype(np.int64),
            "tuned_threshold": float(tuned_threshold),
            "pred_at_tuned": (probs >= float(tuned_threshold)).astype(np.int64),
        }
    )


def format_metric(value):
    if pd.isna(value):
        return "n/a"
    return f"{value:.4f}"


def build_split_assignments(dataset, train_idx, val_idx, test_idx, fold_number):
    def _rows(indices, split_name):
        if len(indices) == 0:
            return pd.DataFrame(columns=["fold", "slide_id", "label", "split"])
        view = dataset.df.iloc[indices][["slide_id", "label"]].copy()
        view["fold"] = fold_number
        view["split"] = split_name
        return view[["fold", "slide_id", "label", "split"]]

    out = pd.concat(
        [
            _rows(train_idx, "train"),
            _rows(val_idx, "val"),
            _rows(test_idx, "test"),
        ],
        ignore_index=True,
    )
    return out


def train_one_fold(dataset, fold_number, train_idx, val_idx, test_idx):
    set_seed(SEED + fold_number)

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        Subset(dataset, test_idx.tolist()),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = CLAM_SB().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    train_labels = dataset.df.iloc[train_idx]["label"].to_numpy()
    criterion = build_criterion(train_labels)

    fold_prefix = f"split_{SPLIT_NAME}_fold_{fold_number}"
    best_model_path = os.path.join(MODEL_DIR, f"best_clam_{fold_prefix}.pth")
    history_path = os.path.join(MODEL_DIR, f"{fold_prefix}_history.csv")
    val_predictions_path = os.path.join(MODEL_DIR, f"{fold_prefix}_val_predictions.csv")
    test_predictions_path = os.path.join(MODEL_DIR, f"{fold_prefix}_test_predictions.csv")
    combined_predictions_path = os.path.join(MODEL_DIR, f"{fold_prefix}_predictions.csv")
    summary_path = os.path.join(MODEL_DIR, f"{fold_prefix}_summary.csv")
    split_assignments_path = os.path.join(MODEL_DIR, f"{fold_prefix}_assignments.csv")

    best_snapshot = None
    last_snapshot = None
    best_predictions_df = pd.DataFrame()
    history = []

    print(f"\n=== Fold {fold_number}/{N_SPLITS}: Stratified 70/10/20 Training ===")
    print(f"Train slides: {len(train_idx)} | Val slides: {len(val_idx)} | Test slides: {len(test_idx)}")

    for epoch in range(EPOCHS):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            decision_threshold=DEFAULT_THRESHOLD,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            decision_threshold=DEFAULT_THRESHOLD,
            sample_indices=val_idx,
        )

        tuned_val = tune_threshold(
            val_metrics["labels"],
            val_metrics["probs"],
            preferred_threshold=DEFAULT_THRESHOLD,
        )

        last_snapshot = {
            "epoch": epoch + 1,
            "val_metrics": val_metrics,
            "tuned_val": tuned_val,
        }

        history.append(
            {
                "fold": fold_number,
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy_at_0_76": train_metrics["accuracy"],
                "train_auc": train_metrics["auc"],
                "train_processed_batches": train_metrics["processed_batches"],
                "train_skipped_batches": train_metrics["skipped_batches"],
                "val_loss": val_metrics["loss"],
                "val_accuracy_at_0_76": val_metrics["accuracy"],
                "val_auc": val_metrics["auc"],
                "val_processed_batches": val_metrics["processed_batches"],
                "val_skipped_batches": val_metrics["skipped_batches"],
                "tuned_threshold": tuned_val["threshold"],
                "tuned_accuracy": tuned_val["accuracy"],
                "tuned_sensitivity": tuned_val["sensitivity"],
                "tuned_specificity": tuned_val["specificity"],
                "tuned_f1": tuned_val["f1"],
                "tuned_youden_j": tuned_val["youden_j"],
            }
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss {format_metric(train_metrics['loss'])} | "
            f"Train AUC {format_metric(train_metrics['auc'])} | "
            f"Val Loss {format_metric(val_metrics['loss'])} | "
            f"Val AUC {format_metric(val_metrics['auc'])} | "
            f"Thr {format_metric(tuned_val['threshold'])} | "
            f"Tuned Acc {format_metric(tuned_val['accuracy'])}"
        )

        if not pd.isna(val_metrics["loss"]) and (
            best_snapshot is None or val_metrics["loss"] < best_snapshot["val_metrics"]["loss"]
        ):
            best_snapshot = last_snapshot
            best_predictions_df = build_prediction_frame(
                dataset,
                val_metrics["indices"],
                val_metrics["labels"],
                val_metrics["probs"],
                "val",
                tuned_val["threshold"],
                fold_number=fold_number,
            )
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated: {best_model_path}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_snapshot is None:
        best_snapshot = last_snapshot
        if best_snapshot is not None:
            best_predictions_df = build_prediction_frame(
                dataset,
                best_snapshot["val_metrics"]["indices"],
                best_snapshot["val_metrics"]["labels"],
                best_snapshot["val_metrics"]["probs"],
                "val",
                best_snapshot["tuned_val"]["threshold"],
                fold_number=fold_number,
            )
        torch.save(model.state_dict(), best_model_path)

    best_predictions_df.to_csv(val_predictions_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)

    tuned_threshold = DEFAULT_THRESHOLD
    if best_snapshot is not None:
        tuned_threshold = float(best_snapshot["tuned_val"]["threshold"])

    best_model = CLAM_SB().to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model.eval()

    test_default = run_epoch(
        best_model,
        test_loader,
        criterion,
        decision_threshold=DEFAULT_THRESHOLD,
        sample_indices=test_idx,
    )
    test_tuned = compute_binary_metrics(
        test_default["labels"],
        test_default["probs"],
        tuned_threshold,
    )

    test_predictions_df = build_prediction_frame(
        dataset,
        test_default["indices"],
        test_default["labels"],
        test_default["probs"],
        "test",
        tuned_threshold,
        fold_number=fold_number,
    )
    test_predictions_df.to_csv(test_predictions_path, index=False)

    combined_predictions = pd.concat([best_predictions_df, test_predictions_df], ignore_index=True)
    combined_predictions.to_csv(combined_predictions_path, index=False)

    if best_snapshot is None:
        val_loss = np.nan
        val_auc = np.nan
        val_default_acc = np.nan
        val_tuned_acc = np.nan
        val_tuned_sens = np.nan
        val_tuned_spec = np.nan
        val_tuned_f1 = np.nan
        val_tuned_youden = np.nan
        best_epoch = np.nan
    else:
        val_loss = best_snapshot["val_metrics"]["loss"]
        val_auc = best_snapshot["val_metrics"]["auc"]
        val_default_acc = best_snapshot["val_metrics"]["accuracy"]
        val_tuned_acc = best_snapshot["tuned_val"]["accuracy"]
        val_tuned_sens = best_snapshot["tuned_val"]["sensitivity"]
        val_tuned_spec = best_snapshot["tuned_val"]["specificity"]
        val_tuned_f1 = best_snapshot["tuned_val"]["f1"]
        val_tuned_youden = best_snapshot["tuned_val"]["youden_j"]
        best_epoch = best_snapshot["epoch"]

    train_size, train_neg, train_pos = split_label_counts(dataset.labels, train_idx)
    val_size, val_neg, val_pos = split_label_counts(dataset.labels, val_idx)
    test_size, test_neg, test_pos = split_label_counts(dataset.labels, test_idx)

    summary = pd.DataFrame(
        [
            {
                "fold": fold_number,
                "split_name": SPLIT_NAME,
                "seed": SEED,
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
                "test_ratio": TEST_RATIO,
                "train_size": train_size,
                "train_neg": train_neg,
                "train_pos": train_pos,
                "val_size": val_size,
                "val_neg": val_neg,
                "val_pos": val_pos,
                "test_size": test_size,
                "test_neg": test_neg,
                "test_pos": test_pos,
                "best_epoch": best_epoch,
                "best_val_loss": val_loss,
                "best_val_auc": val_auc,
                "default_threshold": DEFAULT_THRESHOLD,
                "tuned_threshold": tuned_threshold,
                "val_accuracy_at_default": val_default_acc,
                "val_accuracy_at_tuned": val_tuned_acc,
                "val_sensitivity_at_tuned": val_tuned_sens,
                "val_specificity_at_tuned": val_tuned_spec,
                "val_f1_at_tuned": val_tuned_f1,
                "val_youden_j_at_tuned": val_tuned_youden,
                "test_accuracy_at_default": test_default["accuracy"],
                "test_accuracy_at_tuned": test_tuned["accuracy"],
                "test_sensitivity_at_tuned": test_tuned["sensitivity"],
                "test_specificity_at_tuned": test_tuned["specificity"],
                "test_f1_at_tuned": test_tuned["f1"],
                "test_auc": test_tuned["auc"],
                "model_path": best_model_path,
                "history_path": history_path,
                "split_assignments_path": split_assignments_path,
                "val_predictions_path": val_predictions_path,
                "test_predictions_path": test_predictions_path,
                "combined_predictions_path": combined_predictions_path,
            }
        ]
    )
    summary.to_csv(summary_path, index=False)

    split_assignments = build_split_assignments(dataset, train_idx, val_idx, test_idx, fold_number)
    split_assignments.to_csv(split_assignments_path, index=False)

    print("\n=== 70/10/20 Split Summary ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {format_metric(val_loss)}")
    print(f"Best val AUC: {format_metric(val_auc)}")
    print(f"Tuned threshold (from val): {format_metric(tuned_threshold)}")
    print(f"Test accuracy @ default ({DEFAULT_THRESHOLD}): {format_metric(test_default['accuracy'])}")
    print(f"Test accuracy @ tuned ({tuned_threshold:.2f}): {format_metric(test_tuned['accuracy'])}")
    print(f"Test AUC: {format_metric(test_tuned['auc'])}")

    print(f"\nSaved model: {best_model_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved split assignments: {split_assignments_path}")
    print(f"Saved val predictions: {val_predictions_path}")
    print(f"Saved test predictions: {test_predictions_path}")
    print(f"Saved combined predictions: {combined_predictions_path}")
    print(f"Saved summary: {summary_path}")

    return summary.iloc[0].to_dict(), split_assignments, combined_predictions


def train_five_fold():
    dataset = MILDataset(FEATURE_DIR, LABEL_FILE)
    if len(dataset) == 0:
        raise RuntimeError("No valid training samples found after feature/label filtering.")

    labels = np.asarray(dataset.labels, dtype=np.int64)
    all_indices = np.arange(len(dataset), dtype=int)

    # 20% test per fold from outer stratified 5-fold split.
    outer_test_folds = make_stratified_folds(labels, n_splits=N_SPLITS, seed=SEED)

    # Remaining 80% is split into 70/10 by taking 12.5% val from train+val pool.
    val_ratio_within_train_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

    fold_summaries = []
    all_assignments = []
    all_predictions = []

    for fold_number, test_idx in enumerate(outer_test_folds, start=1):
        train_val_idx = np.setdiff1d(all_indices, test_idx, assume_unique=True)
        train_idx, val_idx = make_stratified_train_val_split(
            indices=train_val_idx,
            labels=labels[train_val_idx],
            val_ratio=val_ratio_within_train_val,
            seed=SEED + fold_number,
        )
        validate_fold_split(train_idx, val_idx, test_idx, len(dataset))

        train_size, train_neg, train_pos = split_label_counts(labels, train_idx)
        val_size, val_neg, val_pos = split_label_counts(labels, val_idx)
        test_size, test_neg, test_pos = split_label_counts(labels, test_idx)

        print(f"\n--- Fold {fold_number}/{N_SPLITS} Split ---")
        print(
            f"train={train_size} (neg={train_neg}, pos={train_pos}) | "
            f"val={val_size} (neg={val_neg}, pos={val_pos}) | "
            f"test={test_size} (neg={test_neg}, pos={test_pos})"
        )

        fold_summary, fold_assignments, fold_predictions = train_one_fold(
            dataset,
            fold_number,
            train_idx,
            val_idx,
            test_idx,
        )
        fold_summaries.append(fold_summary)
        all_assignments.append(fold_assignments)
        all_predictions.append(fold_predictions)

    summary_df = pd.DataFrame(fold_summaries)
    numeric_cols = [
        col
        for col in summary_df.columns
        if col != "fold" and pd.api.types.is_numeric_dtype(summary_df[col])
    ]
    mean_row = {}
    for col in summary_df.columns:
        if col == "fold":
            mean_row[col] = "mean"
        elif col in numeric_cols:
            mean_row[col] = summary_df[col].mean()
        else:
            mean_row[col] = ""

    summary_df = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)

    all_assignments_df = pd.concat(all_assignments, ignore_index=True)
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)

    summary_out = os.path.join(MODEL_DIR, f"split_{SPLIT_NAME}_five_fold_summary.csv")
    assignments_out = os.path.join(MODEL_DIR, f"split_{SPLIT_NAME}_five_fold_assignments.csv")
    predictions_out = os.path.join(MODEL_DIR, f"split_{SPLIT_NAME}_five_fold_predictions.csv")

    summary_df.to_csv(summary_out, index=False)
    all_assignments_df.to_csv(assignments_out, index=False)
    all_predictions_df.to_csv(predictions_out, index=False)

    print("\n=== Combined Fold-Wise Split Outputs ===")
    print(f"Summary: {summary_out}")
    print(f"Assignments: {assignments_out}")
    print(f"Predictions: {predictions_out}")


def main():
    if not np.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0")

    set_seed(SEED)
    train_five_fold()


if __name__ == "__main__":
    main()