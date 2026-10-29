"""
build_graz_index.py -- builds graz_index.csv for pretrain_graz_backbone.py,
from the GRAZPEDWRI-DX layout confirmed on the cluster:

    ~/mediinsight_project/data/GRAZPEDWRI-DX/
        dataset.csv              <- metadata, one row per image, columns include
                                     filestem, patient_id, fracture_visible (1.0 or NaN)
        <filestem>.png           <- images, flat, directly under GRAZPEDWRI-DX/
        yolov5/labels/*.txt      <- YOLO box annotations (not used here -- dataset.csv's
                                     own fracture_visible column is the more reliable source)
        pascalvoc/*.xml          <- Pascal VOC box annotations (not used here either)

Label derivation: `fracture_visible` is 1.0 for a fracture-positive image and
NaN (missing) for a fracture-negative one -- confirmed directly
(dataset.shape = (20327, 17); value_counts: 1.0 -> 13550, NaN -> 6777).
has_fracture = 1 if fracture_visible == 1.0 else 0.

Split: patient-level (not row-level) 80/10/10 train/val/test, seeded for
reproducibility -- the same principle data_utils.py already applies to MURA
(a patient's images must not span train and another split, or evaluation
leaks information about that patient's own images seen during training).
GRAZPEDWRI-DX has multiple studies/images per patient_id, same concern as
MURA.

Usage:
    python3 build_graz_index.py \
        --graz-dir "$HOME/mediinsight_project/data/GRAZPEDWRI-DX" \
        --output-csv "$HOME/mediinsight_project/data/graz_index.csv"
"""

import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graz-dir", required=True,
                     help="Path to the GRAZPEDWRI-DX folder containing dataset.csv and the .png images.")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    # test-frac is whatever remains
    args = ap.parse_args()

    dataset_csv = os.path.join(args.graz_dir, "dataset.csv")
    df = pd.read_csv(dataset_csv)
    print(f"Loaded {dataset_csv}: {len(df)} rows, columns: {list(df.columns)}")

    required = {"filestem", "patient_id", "fracture_visible"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset.csv is missing expected column(s): {missing}")

    df["has_fracture"] = (df["fracture_visible"] == 1.0).astype(int)
    print(f"Label distribution (has_fracture): {df['has_fracture'].value_counts().to_dict()}")

    df["image_path"] = df["filestem"].apply(lambda s: os.path.join(args.graz_dir, f"{s}.png"))

    # Keep only rows whose image file actually exists on disk -- dataset.csv
    # may reference images not present in this particular extraction (e.g.
    # if only some of the images_part{1-4}.zip files were unpacked).
    exists_mask = df["image_path"].apply(os.path.exists)
    n_missing = (~exists_mask).sum()
    if n_missing:
        print(f"WARNING: {n_missing} of {len(df)} rows reference an image file that was not found "
              f"on disk -- dropping these rows. (Check that all images_part{{1-4}}.zip were extracted "
              f"if this number is large.)")
    df = df[exists_mask].reset_index(drop=True)
    print(f"{len(df)} rows remain after checking image files exist.")

    # Patient-level split: shuffle unique patient_ids, then assign every row
    # for a given patient_id entirely to one split.
    rng = np.random.RandomState(args.seed)
    patient_ids = df["patient_id"].unique()
    rng.shuffle(patient_ids)

    n_patients = len(patient_ids)
    n_train = int(args.train_frac * n_patients)
    n_val = int(args.val_frac * n_patients)

    train_patients = set(patient_ids[:n_train])
    val_patients = set(patient_ids[n_train:n_train + n_val])
    test_patients = set(patient_ids[n_train + n_val:])

    def split_for(pid):
        if pid in train_patients:
            return "train"
        elif pid in val_patients:
            return "val"
        else:
            return "test"

    df["split"] = df["patient_id"].apply(split_for)

    out = df[["image_path", "has_fracture", "split"]]
    out.to_csv(args.output_csv, index=False)

    print(f"\nWrote {args.output_csv}")
    print(f"Patients: {n_patients} total -> {len(train_patients)} train / "
          f"{len(val_patients)} val / {len(test_patients)} test")
    print("Row counts per split:")
    print(out["split"].value_counts())
    print("\nhas_fracture distribution per split:")
    print(out.groupby("split")["has_fracture"].value_counts())


if __name__ == "__main__":
    main()
