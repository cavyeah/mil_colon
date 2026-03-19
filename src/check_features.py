import argparse
import csv
import os
import re
import sys

import numpy as np
import torch

FEATURE_ROOT_DEFAULT = "data/features"
PATCH_ROOT_DEFAULT = "data/patches/tcga_coad"
REPORT_DEFAULT = "data/features/feature_check_report.csv"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
COORD_PATTERN = re.compile(r"(-?\d+)[_x,](-?\d+)$")
FLOAT_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)


def natural_sort_key(name: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


def list_patch_files(patch_dir: str):
    files = []
    with os.scandir(patch_dir) as entries:
        for entry in entries:
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in IMG_EXTS:
                files.append(entry.name)
    return files


def sorted_coords_rows(coords: np.ndarray):
    order = np.lexsort((coords[:, 0], coords[:, 1]))
    return coords[order]


def same_coord_set(a: np.ndarray, b: np.ndarray):
    if len(a) != len(b):
        return False
    return np.array_equal(sorted_coords_rows(a), sorted_coords_rows(b))


def parse_filename_coords(patch_files):
    # expects coords at end of stem, e.g. slide_1024_2048.png
    coords = []
    for fname in patch_files:
        stem = os.path.splitext(fname)[0]
        m = COORD_PATTERN.search(stem)
        if m is None:
            return None
        coords.append((int(m.group(1)), int(m.group(2))))
    coords_np = np.asarray(coords, dtype=np.int64)
    if len(coords_np) > 1:
        coords_np = sorted_coords_rows(coords_np)
    return coords_np


def _to_numpy_coords(coords_obj):
    if coords_obj is None:
        return None
    if torch.is_tensor(coords_obj):
        return coords_obj.detach().cpu().numpy()
    return np.asarray(coords_obj)


def check_one(feature_path: str, patch_root: str, expected_dim: int):
    slide_id = os.path.splitext(os.path.basename(feature_path))[0]

    errors = []
    warnings = []

    n_patches = -1
    feat_dim = -1
    patch_count = -1
    nan_count = -1
    inf_count = -1

    features = None
    coords_np = None

    # --- load feature file
    try:
        obj = torch.load(feature_path, map_location="cpu")
    except Exception as e:
        errors.append(f"torch.load failed: {e}")
        obj = None

    if obj is not None:
        if torch.is_tensor(obj):
            # legacy format
            features = obj
            warnings.append("legacy tensor-only format (missing coords key)")
        elif isinstance(obj, dict):
            features = obj.get("features", obj.get("feats"))
            coords_np = _to_numpy_coords(obj.get("coords"))
        else:
            errors.append(f"unsupported feature file type: {type(obj).__name__}")

    # --- validate features
    if features is None or not torch.is_tensor(features):
        errors.append("missing/invalid features tensor")
    else:
        if features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)

        if features.ndim != 2:
            errors.append(f"features must be 2D, got shape {tuple(features.shape)}")
        else:
            n_patches = int(features.shape[0])
            feat_dim = int(features.shape[1])

            if n_patches <= 0:
                errors.append("features tensor is empty")

            if expected_dim > 0 and feat_dim != expected_dim:
                warnings.append(f"feature dim={feat_dim}, expected={expected_dim}")

            if features.dtype not in FLOAT_DTYPES:
                warnings.append(f"non-float feature dtype: {features.dtype}")
                nan_count = 0
                inf_count = 0
            else:
                # Fast path: only compute nan/inf counts if non-finite values exist.
                if bool(torch.isfinite(features).all()):
                    nan_count = 0
                    inf_count = 0
                else:
                    nan_count = int(torch.isnan(features).sum().item())
                    inf_count = int(torch.isinf(features).sum().item())
                    errors.append(f"non-finite values found (nan={nan_count}, inf={inf_count})")

    # --- validate coords in feature file
    if coords_np is None:
        errors.append("missing coords in feature file")
    else:
        if coords_np.ndim != 2 or coords_np.shape[1] < 2:
            errors.append(f"coords must be Nx2+, got shape {coords_np.shape}")
            coords_np = None
        else:
            coords_np = coords_np[:, :2]
            if not np.issubdtype(coords_np.dtype, np.number):
                errors.append(f"coords must be numeric, got dtype {coords_np.dtype}")
                coords_np = None
            elif not np.isfinite(coords_np).all():
                errors.append("coords contain non-finite values")
                coords_np = None
            else:
                coords_np = coords_np.astype(np.int64, copy=False)

            if coords_np is not None and n_patches >= 0 and len(coords_np) != n_patches:
                errors.append(f"coords/features length mismatch: coords={len(coords_np)}, features={n_patches}")

            if coords_np is not None and len(coords_np) > 0:
                uniq = np.unique(coords_np, axis=0)
                if len(uniq) != len(coords_np):
                    errors.append("duplicate coordinates detected")

    # --- compare with patch folder
    patch_dir = os.path.join(patch_root, slide_id)
    if not os.path.isdir(patch_dir):
        warnings.append(f"patch folder missing: {patch_dir}")
        patch_files = []
    else:
        patch_files = list_patch_files(patch_dir)
        patch_count = len(patch_files)

        if patch_count == 0:
            errors.append("no patch images found in patch folder")

        if n_patches >= 0 and patch_count != n_patches:
            errors.append(f"patch/features count mismatch: patches={patch_count}, features={n_patches}")

        coord_file = os.path.join(patch_dir, "coords.npy")
        if coords_np is not None and os.path.exists(coord_file):
            try:
                c = np.load(coord_file)
                if c.ndim == 2 and c.shape[1] >= 2:
                    c = c[:, :2].astype(np.int64)
                    if len(c) == len(coords_np) and not np.array_equal(c, coords_np):
                        if same_coord_set(c, coords_np):
                            warnings.append("coords order differs from patch-folder coords.npy")
                        else:
                            warnings.append("coords values differ from patch-folder coords.npy")
                else:
                    warnings.append("coords.npy has invalid shape")
            except Exception as e:
                warnings.append(f"failed to read coords.npy: {e}")

        if coords_np is not None and patch_files:
            parsed = parse_filename_coords(patch_files)
            if parsed is not None and len(parsed) == len(coords_np):
                if not np.array_equal(parsed, coords_np):
                    if same_coord_set(parsed, coords_np):
                        warnings.append("coords order differs from filename-parsed row-major coordinates")
                    else:
                        warnings.append("coords values differ from filename-parsed coordinates")

    status = "PASS" if len(errors) == 0 else "FAIL"

    row = {
        "slide_id": slide_id,
        "status": status,
        "n_patches": n_patches,
        "feat_dim": feat_dim,
        "patch_count": patch_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": " | ".join(errors),
        "warnings": " | ".join(warnings),
        "feature_path": feature_path,
    }
    return row


def main():
    parser = argparse.ArgumentParser(description="Validate extracted feature files for heatmap alignment.")
    parser.add_argument("--feature-root", default=FEATURE_ROOT_DEFAULT)
    parser.add_argument("--patch-root", default=PATCH_ROOT_DEFAULT)
    parser.add_argument("--slide-id", default=None, help="Check one slide id (without .pt)")
    parser.add_argument("--expected-dim", type=int, default=2048, help="Set <=0 to disable")
    parser.add_argument("--report", default=REPORT_DEFAULT)
    parser.add_argument("--strict", action="store_true", help="Exit code 1 if any file fails")
    args = parser.parse_args()

    if not os.path.isdir(args.feature_root):
        print(f"[ERROR] feature root not found: {args.feature_root}")
        sys.exit(1)

    if args.slide_id:
        feature_files = [os.path.join(args.feature_root, f"{args.slide_id}.pt")]
        if not os.path.exists(feature_files[0]):
            print(f"[ERROR] feature file not found: {feature_files[0]}")
            sys.exit(1)
    else:
        feature_files = []
        with os.scandir(args.feature_root) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".pt"):
                    feature_files.append(os.path.join(args.feature_root, entry.name))
        feature_files.sort(key=lambda p: natural_sort_key(os.path.basename(p)))

    if len(feature_files) == 0:
        print("[ERROR] no feature files found")
        sys.exit(1)

    rows = []
    for fp in feature_files:
        row = check_one(fp, args.patch_root, args.expected_dim)
        rows.append(row)

        print(
            f"[{row['status']}] {row['slide_id']} | "
            f"N={row['n_patches']} D={row['feat_dim']} | "
            f"errors={row['n_errors']} warnings={row['n_warnings']}"
        )
        if row["errors"]:
            print(f"  errors:   {row['errors']}")
        if row["warnings"]:
            print(f"  warnings: {row['warnings']}")

    report_dir = os.path.dirname(args.report)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    fieldnames = [
        "slide_id", "status", "n_patches", "feat_dim", "patch_count",
        "nan_count", "inf_count", "n_errors", "n_warnings",
        "errors", "warnings", "feature_path"
    ]
    with open(args.report, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_total = len(rows)
    n_fail = sum(r["status"] == "FAIL" for r in rows)
    n_pass = n_total - n_fail

    print(f"\nDone. PASS={n_pass} FAIL={n_fail} TOTAL={n_total}")
    print(f"Report: {args.report}")

    if args.strict and n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()