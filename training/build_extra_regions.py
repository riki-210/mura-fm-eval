"""build_extra_regions.py -- index the MURA body parts that are NOT in mura_ehr_index.csv (finger, forearm, humerus, wrist),
for use as EXTRA TRAINING DATA only (never val/test). Scans a MURA folder tree with study folders like
.../XR_WRIST/patient00012/study1_positive/image1.png ; the label comes from the study folder name (positive -> 1)."""
import argparse, glob, os, re
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--mura-root", required=True, help="folder that contains the MURA train/valid trees (searched recursively)")
ap.add_argument("--index-csv", required=True, help="the existing mura_ehr_index.csv (its regions are skipped)")
ap.add_argument("--out", required=True)
args = ap.parse_args()

have = set(pd.read_csv(args.index_csv)["region"].unique())
files = []
for ext in ("png", "jpg", "jpeg"):
    files += glob.glob(os.path.join(args.mura_root, "**", f"XR_*", "patient*", "study*", f"*.{ext}"), recursive=True)
rows = []
for f in files:
    parts = f.replace("\\", "/").split("/")
    reg = next((p for p in parts if re.fullmatch(r"XR_[A-Z]+", p)), None)
    pat = next((p for p in parts if re.fullmatch(r"patient\d+", p)), None)
    study = next((p for p in parts if re.fullmatch(r"study\d+_(positive|negative)", p)), None)
    if reg is None or pat is None or study is None or reg in have:
        continue
    rows.append({"image_path": f, "patient_id": f"{reg}_{pat}", "region": reg,
                 "fracture_label": 1 if study.endswith("positive") else 0, "split": "train"})
df = pd.DataFrame(rows).drop_duplicates("image_path")
df.to_csv(args.out, index=False)
print(f"regions already in index (skipped): {sorted(have)}")
print(f"wrote {len(df)} images / {df['patient_id'].nunique()} patients to {args.out}")
print(df.groupby("region")["fracture_label"].agg(["count", "mean"]).round(3).to_string())
