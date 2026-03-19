import hashlib
import os

import numpy as np
import pandas as pd
import torch

from clam_model import CLAM_SB

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(ROOT_DIR, "data", "features")
LABEL_FILE = os.path.join(ROOT_DIR, "data", "labels.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "src", "models", "five_fold")

THRESHOLD_SUMMARY_FILE = os.path.join(MODEL_DIR, "five_fold_threshold_summary.csv")
FOLD_SUMMARY_FILE = os.path.join(MODEL_DIR, "five_fold_summary.csv")
PREDICTIONS_FILE = os.path.join(MODEL_DIR, "ensemble_predictions.csv")
METRICS_FILE = os.path.join(MODEL_DIR, "ensemble_metrics.csv")

NUM_FOLDS = 5
DEFAULT_THRESHOLD = 0.66
MAX_PATCHES = 14000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODEL_DIR, exist_ok=True)


def format_metric(value):
    if pd.isna(value):
        return "n/a"
    return f"{value:.4f}"


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


def compute_metrics(labels, probs, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float32)

    if labels.size == 0:
        return {
            "threshold": float(threshold),
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "specificity": np.nan,
            "f1": np.nan,
            "auc": np.nan,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "n_slides": 0,
        }

    preds = (probs >= threshold).astype(np.int64)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    total = int(labels.size)
    accuracy = (tp + tn) / total if total > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    if pd.isna(precision) or pd.isna(recall) or (precision + recall) == 0:
        f1 = np.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy) if not pd.isna(accuracy) else np.nan,
        "precision": float(precision) if not pd.isna(precision) else np.nan,
        "recall": float(recall) if not pd.isna(recall) else np.nan,
        "specificity": float(specificity) if not pd.isna(specificity) else np.nan,
        "f1": float(f1) if not pd.isna(f1) else np.nan,
        "auc": compute_auc(labels, probs),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_slides": total,
    }


def resolve_threshold():
    if os.path.exists(THRESHOLD_SUMMARY_FILE):
        try:
            df = pd.read_csv(THRESHOLD_SUMMARY_FILE)
            if "tuned_threshold" in df.columns and not df.empty:
                if "scope" in df.columns:
                    oof_df = df[df["scope"].astype(str).str.lower() == "out_of_fold"]
                    if not oof_df.empty:
                        value = float(oof_df["tuned_threshold"].iloc[0])
                        if np.isfinite(value):
                            return value, "five_fold_threshold_summary.csv"

                tuned_values = pd.to_numeric(df["tuned_threshold"], errors="coerce").dropna()
                if not tuned_values.empty:
                    value = float(tuned_values.iloc[0])
                    if np.isfinite(value):
                        return value, "five_fold_threshold_summary.csv"
        except Exception:
            pass

    if os.path.exists(FOLD_SUMMARY_FILE):
        try:
            df = pd.read_csv(FOLD_SUMMARY_FILE)
            if "best_threshold" in df.columns:
                tuned_values = pd.to_numeric(df["best_threshold"], errors="coerce").dropna()
                if not tuned_values.empty:
                    value = float(tuned_values.mean())
                    if np.isfinite(value):
                        return value, "five_fold_summary.csv(mean best_threshold)"
        except Exception:
            pass

    return DEFAULT_THRESHOLD, "default"


def normalize_state_dict(loaded_obj):
    if isinstance(loaded_obj, dict) and "state_dict" in loaded_obj:
        state_dict = loaded_obj["state_dict"]
    else:
        state_dict = loaded_obj

    if not isinstance(state_dict, dict):
        raise TypeError("Unsupported checkpoint format.")

    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    return state_dict


def load_models():
    models = []

    for fold_idx in range(1, NUM_FOLDS + 1):
        model_path = os.path.join(MODEL_DIR, f"best_clam_fold_{fold_idx}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing fold model: {model_path}")

        model = CLAM_SB().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = normalize_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        models.append(model)

    return models


def subsample_patches(features, slide_id):
    if MAX_PATCHES is None or features.shape[0] <= MAX_PATCHES:
        return features

    digest = hashlib.sha256(str(slide_id).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(features.shape[0], size=MAX_PATCHES, replace=False))
    indices = torch.from_numpy(indices).long()

    return features[indices]


def load_feature_tensor(slide_id):
    slide_name = str(slide_id).strip()
    if slide_name.lower().endswith(".pt"):
        slide_name = slide_name[:-3]

    feature_path = os.path.join(FEATURE_DIR, slide_name + ".pt")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Missing feature file: {feature_path}")

    loaded = torch.load(feature_path, map_location="cpu")

    if isinstance(loaded, dict):
        if "features" in loaded:
            features = loaded["features"]
        elif "feats" in loaded:
            features = loaded["feats"]
        else:
            raise ValueError(f"Unsupported feature dict format: {feature_path}")
    else:
        features = loaded

    if not torch.is_tensor(features):
        raise TypeError(f"Feature file does not contain a tensor: {feature_path}")

    features = features.float()
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if features.ndim == 3 and features.shape[0] == 1:
        features = features.squeeze(0)

    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D feature tensor, got shape {tuple(features.shape)} for {feature_path}"
        )

    if features.shape[0] == 0:
        raise ValueError(f"Empty feature tensor: {feature_path}")

    original_patch_count = int(features.shape[0])
    features = subsample_patches(features, slide_name)
    used_patch_count = int(features.shape[0])

    return features, original_patch_count, used_patch_count


def predict_slide(models, features):
    features = features.to(device, non_blocking=True)
    fold_probs = []

    with torch.inference_mode():
        for model in models:
            logits, _ = model(features)
            prob = float(torch.sigmoid(logits).reshape(-1)[0].detach().cpu().item())

            if not np.isfinite(prob):
                raise ValueError("Non-finite probability produced during inference.")

            fold_probs.append(prob)

    fold_probs = np.asarray(fold_probs, dtype=np.float32)
    ensemble_prob = float(fold_probs.mean())
    prob_std = float(fold_probs.std())

    return ensemble_prob, prob_std, fold_probs.tolist()


def main():
    if not os.path.exists(LABEL_FILE):
        raise FileNotFoundError(f"Missing label file: {LABEL_FILE}")

    df = pd.read_csv(LABEL_FILE)
    required_columns = {"slide_id", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"labels.csv is missing required columns: {sorted(missing_columns)}")

    df = df[["slide_id", "label"]].dropna().copy()
    df["label"] = df["label"].astype(int)

    unique_labels = set(df["label"].unique().tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Only binary labels 0/1 are supported. Found: {sorted(unique_labels)}")

    models = load_models()
    threshold, threshold_source = resolve_threshold()

    print(f"Loaded {len(models)} fold models")
    print(f"Device: {device}")
    print(f"Threshold: {threshold:.4f} ({threshold_source})")
    print("\nSlide Predictions\n")

    rows = []

    for _, row in df.iterrows():
        slide_id = str(row["slide_id"]).strip()
        label = int(row["label"])

        features, original_patch_count, used_patch_count = load_feature_tensor(slide_id)
        ensemble_prob, prob_std, fold_probs = predict_slide(models, features)

        pred = int(ensemble_prob >= threshold)

        result = {
            "slide_id": slide_id,
            "label": label,
            "original_num_patches": original_patch_count,
            "num_patches_used": used_patch_count,
            "ensemble_prob": ensemble_prob,
            "ensemble_prob_std": prob_std,
            "threshold": float(threshold),
            "threshold_source": threshold_source,
            "pred": pred,
        }

        for fold_idx, fold_prob in enumerate(fold_probs, start=1):
            result[f"fold_{fold_idx}_prob"] = float(fold_prob)

        rows.append(result)

        print(
            f"{slide_id} | prob={ensemble_prob:.4f} | std={prob_std:.4f} | "
            f"pred={pred} | label={label}"
        )

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(PREDICTIONS_FILE, index=False)

    metrics = compute_metrics(
        labels=pred_df["label"].to_numpy(),
        probs=pred_df["ensemble_prob"].to_numpy(),
        threshold=threshold,
    )
    metrics["threshold_source"] = threshold_source

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(METRICS_FILE, index=False)

    print("\nEvaluation Metrics\n")
    print(f"Accuracy    : {format_metric(metrics['accuracy'])}")
    print(f"Precision   : {format_metric(metrics['precision'])}")
    print(f"Recall      : {format_metric(metrics['recall'])}")
    print(f"Specificity : {format_metric(metrics['specificity'])}")
    print(f"F1 Score    : {format_metric(metrics['f1'])}")
    print(f"AUC         : {format_metric(metrics['auc'])}")

    print(f"\nPredictions saved to: {PREDICTIONS_FILE}")
    print(f"Metrics saved to: {METRICS_FILE}")
    print(
        "\nNote: if labels.csv is the same cohort used during 5-fold training, "
        "these ensemble metrics are optimistic. Use an external test set for final reporting."
    )


if __name__ == "__main__":
    main()