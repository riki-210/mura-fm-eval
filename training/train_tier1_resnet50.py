"""
train_tier1_resnet50.py -- Tier 1 region classifier (elbow / hand / shoulder),
ResNet50 backbone, per BoneFracture_Tier2_FINAL.pdf: ImageNet-pretrained,
Adam lr=1e-4, batch=32, early stopping patience=5.

Usage:
    python train_tier1_resnet50.py \
        --mura-root /path/to/MURA-v1.1 \
        --index-csv mura_ehr_index.csv \
        --output-dir results/tier1

If --index-csv doesn't exist yet, this script builds it (calls
data_utils.index_mura + make_patient_level_split) and saves it, so you can
run this script directly on a fresh MURA download without a separate
indexing step.
"""

import argparse
import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from data_utils import index_mura, make_patient_level_split, print_split_summary, build_tf_dataset, REGIONS
from models import build_resnet50_region_classifier


def get_or_build_index(mura_root: str, index_csv: str) -> pd.DataFrame:
    if os.path.exists(index_csv):
        print(f"Loading existing index: {index_csv}")
        return pd.read_csv(index_csv)
    print(f"No existing index found -- building from {mura_root} (this walks the whole dataset once)...")
    df = index_mura(mura_root)
    df = make_patient_level_split(df)
    df.to_csv(index_csv, index=False)
    print(f"Index written to {index_csv}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mura-root", required=True)
    ap.add_argument("--index-csv", default="mura_ehr_index.csv")
    ap.add_argument("--output-dir", default="results/tier1")
    ap.add_argument("--epochs", type=int, default=100, help="Max epochs (early stopping patience=5 will typically stop sooner).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--no-enhancement", action="store_true",
                     help="Skip the 4-stage enhancement pipeline (debug / ablation use).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = get_or_build_index(args.mura_root, args.index_csv)
    print_split_summary(df)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    apply_enh = not args.no_enhancement
    train_ds = build_tf_dataset(train_df, label_col="region_label", batch_size=args.batch_size,
                                 augment=True, apply_enhancement=apply_enh, shuffle=True, img_size=args.img_size)
    val_ds = build_tf_dataset(val_df, label_col="region_label", batch_size=args.batch_size,
                               augment=False, apply_enhancement=apply_enh, shuffle=False, img_size=args.img_size)
    test_ds = build_tf_dataset(test_df, label_col="region_label", batch_size=args.batch_size,
                                augment=False, apply_enhancement=apply_enh, shuffle=False, img_size=args.img_size)

    model = build_resnet50_region_classifier(num_classes=len(REGIONS), img_size=args.img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ckpt_path = os.path.join(args.output_dir, "tier1_resnet50_best.h5")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.CSVLogger(os.path.join(args.output_dir, "tier1_training_log.csv")),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTier 1 (region classifier) TEST accuracy: {test_acc:.4f}")

    summary = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "epochs_trained": len(history.history["loss"]),
        "regions": REGIONS,
    }
    with open(os.path.join(args.output_dir, "tier1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {os.path.join(args.output_dir, 'tier1_summary.json')}")


if __name__ == "__main__":
    main()
