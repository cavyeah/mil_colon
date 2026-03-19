import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from clam_model import CLAM_SB


FEATURE_DIR = "data/features"
LABEL_FILE = "data/labels.csv"
MODEL_PATH = "src/models/best_clam_model_topk.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MILDataset(Dataset):

    def __init__(self, feature_dir, csv_file):

        self.feature_dir = feature_dir
        self.df = pd.read_csv(csv_file)

        self.slides = self.df["slide_id"].tolist()
        self.labels = self.df["label"].tolist()

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):

        slide = self.slides[idx]
        label = self.labels[idx]

        path = os.path.join(self.feature_dir, slide + ".pt")

        features = torch.load(path, map_location="cpu").float()
        features = torch.nan_to_num(features)

        return features, torch.tensor(label, dtype=torch.float32), slide


def evaluate():

    dataset = MILDataset(FEATURE_DIR, LABEL_FILE)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    model = CLAM_SB().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    probs = []
    preds = []
    labels = []
    slides = []

    print("\nSlide Predictions\n")

    for feats, label, slide_id in loader:

        feats = feats.squeeze(0).to(device)
        label = label.item()

        with torch.no_grad():
            out, _ = model(feats)

        prob = torch.sigmoid(out).item()

        pred = 1 if prob >= 0.76 else 0

        print(slide_id[0], "| prob=", round(prob,4), "| pred=", pred, "| label=", label)

        probs.append(prob)
        preds.append(pred)
        labels.append(label)
        slides.append(slide_id[0])

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)

    df = pd.DataFrame({
        "slide": slides,
        "prob": probs,
        "pred": preds,
        "label": labels
    })

    df.to_csv("predictions.csv", index=False)

    print("\nEvaluation Metrics\n")

    print("Accuracy  :", round(acc,4))
    print("Precision :", round(prec,4))
    print("Recall    :", round(rec,4))
    print("F1 Score  :", round(f1,4))
    print("AUC       :", round(auc,4))


if __name__ == "__main__":
    evaluate()