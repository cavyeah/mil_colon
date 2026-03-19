import os
import pandas as pd

FEATURE_DIR = "data/features"
OUTPUT_FILE = "data/labels.csv"

rows = []

for f in os.listdir(FEATURE_DIR):

    if not f.endswith(".pt"):
        continue

    slide_id = f.replace(".pt", "")

    # Extract TCGA sample code
    parts = slide_id.split("-")

    if len(parts) < 4:
        print("Skipping unknown slide:", slide_id)
        continue

    sample_code = parts[3]  # e.g. 01Z, 11A

    if sample_code.startswith("01"):
        label = 1   # tumor

    elif sample_code.startswith("11"):
        label = 0   # normal

    else:
        print("Skipping unknown slide type:", slide_id)
        continue

    rows.append({
        "slide_id": slide_id,
        "label": label
    })


df = pd.DataFrame(rows)

df = df.sort_values("slide_id")

df.to_csv(OUTPUT_FILE, index=False)

print("labels.csv regenerated successfully")
print("Total slides:", len(df))
print("Tumor slides:", sum(df["label"] == 1))
print("Normal slides:", sum(df["label"] == 0))