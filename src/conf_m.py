import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

PRED_FILE = "predictions.csv"

df = pd.read_csv(PRED_FILE)

labels = df["label"]
preds = df["pred"]

cm = confusion_matrix(labels, preds)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal","Tumor"],
    yticklabels=["Normal","Tumor"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()