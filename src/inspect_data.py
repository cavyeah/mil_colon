import torch
import os

# =========================
# 🔧 AUTO PICK A FILE (NO MANUAL NAME NEEDED)
# =========================

FEATURE_DIR = "features"
MODEL_PATH = "models/five_fold/best_fold_0.pt"

# pick first .pt file automatically
files = [f for f in os.listdir(FEATURE_DIR) if f.endswith(".pt")]

if len(files) == 0:
    print("❌ No .pt files found in features folder")
    exit()

FEATURE_PATH = os.path.join(FEATURE_DIR, files[0])

print(f"\nUsing feature file: {FEATURE_PATH}")

# =========================
# 📦 LOAD FEATURE FILE
# =========================

print("\n--- LOADING FEATURE FILE ---")
data = torch.load(FEATURE_PATH, map_location='cpu')

print("Type of data:", type(data))

# Case 1: Dictionary
if isinstance(data, dict):
    print("Keys:", data.keys())
    
    if 'features' in data:
        features = data['features']
        print("Feature shape:", features.shape)
    else:
        print("⚠️ No 'features' key found!")
        exit()

    if 'coords' in data:
        coords = data['coords']
        print("Coords shape:", coords.shape)
    else:
        coords = None
        print("⚠️ No coordinates found")

# Case 2: Tensor
elif isinstance(data, torch.Tensor):
    features = data
    coords = None
    print("Tensor shape:", features.shape)

else:
    print("❌ Unknown data format!")
    exit()

# =========================
# 🧠 LOAD MODEL
# =========================

print("\n--- LOADING MODEL ---")

try:
    model = torch.load(MODEL_PATH, map_location='cpu')
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    exit()

# =========================
# 🔍 RUN MODEL (GET ATTENTION)
# =========================

print("\n--- RUNNING MODEL ---")

features = features.float()

try:
    with torch.no_grad():
        logits, Y_prob, Y_hat, A, _ = model(features)
except:
    print("Retrying with batch dimension...")
    with torch.no_grad():
        logits, Y_prob, Y_hat, A, _ = model(features.unsqueeze(0))

print("\n--- OUTPUT ---")
print("Logits shape:", logits.shape)
print("Probabilities shape:", Y_prob.shape)
print("Prediction:", Y_hat)
print("Attention shape:", A.shape)

print("\n--- DONE ---")