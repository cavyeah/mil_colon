"""
test_run_full_inference.py
--------------------------
Quick smoke-test for the run_full_inference() function.
Run from the MIL project root:
    python test_run_full_inference.py
"""

import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import run_full_inference

# ── Locate model checkpoints ───────────────────────────────────────────────────
model_paths = sorted(glob.glob(os.path.join("models", "five_fold", "best_fold_*.pt")))
print(f"Found {len(model_paths)} fold model(s):")
for p in model_paths:
    print(f"  {p}")

if not model_paths:
    print("ERROR: No model checkpoints found. Train first with train_clam_5fold.py")
    sys.exit(1)

# ── Locate a feature file ──────────────────────────────────────────────────────
pt_files = sorted(glob.glob(os.path.join("features", "*.pt")))
if not pt_files:
    print("ERROR: No .pt files found in features/")
    sys.exit(1)

pt_path = pt_files[0]
print(f"\nSlide: {pt_path}\n")

# ── Call run_full_inference ────────────────────────────────────────────────────
print("Running run_full_inference() ...")
print("-" * 50)

result = run_full_inference(pt_path, model_paths)

# ── Print results ──────────────────────────────────────────────────────────────
print("\n=== RESULTS ===")
print(f"  probability   : {result['probability']:.4f}")
print(f"  prediction    : {result['prediction']}")
print(f"  label         : {result['label']}")
print(f"  confidence    : {result['confidence']:.4f}  (distance from threshold 0.276)")
print(f"  n_patches     : {result['n_patches']:,}")
print(f"  has_coords    : {result['has_coords']}")
print(f"  attention     : shape={result['attention'].shape}  dtype={result['attention'].dtype}")

if result["has_coords"]:
    h8  = result["heatmap_8x8"]
    h16 = result["heatmap_16x16"]
    print(f"  heatmap_8x8   : shape={h8.shape}  min={h8.min():.3f}  max={h8.max():.3f}")
    print(f"  heatmap_16x16 : shape={h16.shape}  min={h16.min():.3f}  max={h16.max():.3f}")
else:
    print("  heatmap_8x8   : None (no coords in feature file)")
    print("  heatmap_16x16 : None (no coords in feature file)")

print("\nAll returned keys:", sorted(result.keys()))

# ── Type assertions ────────────────────────────────────────────────────────────
import numpy as np

assert isinstance(result["probability"], float),   "probability must be float"
assert result["prediction"] in (0, 1),              "prediction must be 0 or 1"
assert isinstance(result["label"], str),            "label must be str"
assert isinstance(result["confidence"], float),     "confidence must be float"
assert isinstance(result["n_patches"], int),        "n_patches must be int"
assert isinstance(result["attention"], np.ndarray), "attention must be np.ndarray"
if result["has_coords"]:
    assert result["heatmap_8x8"].shape  == (8, 8),  "heatmap_8x8 shape mismatch"
    assert result["heatmap_16x16"].shape == (16, 16),"heatmap_16x16 shape mismatch"
    assert result["heatmap_8x8"].min()  >= 0.0
    assert result["heatmap_8x8"].max()  <= 1.0

print("\nAll assertions PASSED.")
print("run_full_inference() is ready for Streamlit integration.")
