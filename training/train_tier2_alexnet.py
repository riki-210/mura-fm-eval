"""
train_tier2_alexnet.py -- Tier 2 region-specific binary fracture classifier
(AlexNet, optionally with a PCA front-end -- see pca_utils.py for the
interpretation note on what "AlexNet + PCA" means here), for ONE region at
a time (elbow, hand, or shoulder), per BoneFracture_Tier2_FINAL.pdf.

This script trains a single configuration; run_ablation_5config.py sweeps
the 5-config hyperparameter grid (dropout x L2 x PCA on/off) on the shoulder
subset specifically, calling into the same functions this script uses.

Usage (plain AlexNet, no PCA):
    python train_tier2_alexnet.py --index-csv mura_ehr_index.csv \
        --region XR_SHOULDER --output-dir results/tier2_shoulder_plain \
        --dropout 0.5 --l2 0.0

Usage (AlexNet + PCA):
    python train_tier2_alexnet.py --index-csv mura_ehr_index.csv \
        --region XR_SHOULDER --output-dir results/tier2_shoulder_pca \
        --dropout 0.8 --l2 1e-4 --use-pca
"""

import argparse
import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from data_utils import build_tf_dataset, IMG_SIZE
from models import build_alexnet, build_alexnet_pca_head
from pca_utils import fit_pca, transform_image, save_pca, load_pca


def build_pca_datasets(train_df, val_df, test_df, pca, apply_enhancement, batch_size):
    """
    Pre-computes PCA features for every image (train/val/test) up front --
    this is a one-off CPU pass, not done inside the tf.data pipeline, since
    sklearn's IncrementalPCA doesn't run inside a tf graph. Returns
    tf.data.Dataset objects yielding (pca_vector, label) batches.
    """
    def featurize(df):
        X = np.stack([transform_image(pca, p, apply_enhancement) for p in df["image_path"]])
        y = df["fracture_label"].values.astype(np.int64)
        return X, y

    X_train, y_train = featurize(train_df)
    X_val, y_val = featurize(val_df)
    X_test, y_test = featurize(test_df)

    def make_ds(X, y, shuffle):
        ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X), seed=42)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return make_ds(X_train, y_train, True), make_ds(X_val, y_val, False), make_ds(X_test, y_test, False)


def train_one_config(train_df, val_df, test_df, region: str, dropout: float, l2_reg: float,
                      use_pca: bool, output_dir: str, img_size: int = IMG_SIZE,
                      batch_size: int = 32, epochs: int = 100, apply_enhancement: bool = True,
                      pca_cache_path: str = None) -> dict:
    """
    Trains one (dropout, l2, use_pca) configuration for one region and
    returns a metrics dict. Used both by the single-config CLI below and by
    run_ablation_5config.py's grid loop.
    """
    os.makedirs(output_dir, exist_ok=True)

    if use_pca:
        if pca_cache_path and os.path.exists(pca_cache_path):
            print(f"Loading cached PCA: {pca_cache_path}")
            pca = load_pca(pca_cache_path)
        else:
            print(f"Fitting PCA on {len(train_df)} training images ({region})...")
            paths = train_df["image_path"].tolist()
            if len(paths) > 2000:
                rng = np.random.RandomState(42)
                paths = list(rng.choice(paths, size=2000, replace=False))
            pca = fit_pca(paths, apply_enhancement=apply_enhancement)
            if pca_cache_path:
                save_pca(pca, pca_cache_path)

        train_ds, val_ds, test_ds = build_pca_datasets(
            train_df, val_df, test_df, pca, apply_enhancement, batch_size
        )
        model = build_alexnet_pca_head(
            n_pca_components=pca.n_components_, num_classes=2, dropout=dropout, l2_reg=l2_reg
        )
    else:
        train_ds = build_tf_dataset(train_df, "fracture_label", batch_size, augment=True,
                                     apply_enhancement=apply_enhancement, shuffle=True, img_size=img_size)
        val_ds = build_tf_dataset(val_df, "fracture_label", batch_size, augment=False,
                                   apply_enhancement=apply_enhancement, shuffle=False, img_size=img_size)
        test_ds = build_tf_dataset(test_df, "fracture_label", batch_size, augment=False,
                                    apply_enhancement=apply_enhancement, shuffle=False, img_size=img_size)
        model = build_alexnet(img_size=img_size, num_classes=2, dropout=dropout, l2_reg=l2_reg,
                               input_channels=3)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_path = os.path.join(output_dir, "best_model.h5")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.CSVLogger(os.path.join(output_dir, "training_log.csv")),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, batch_size=batch_size)
    test_loss, test_acc = model.evaluate(test_ds)

    result = {
        "region": region,
        "dropout": dropout,
        "l2_reg": l2_reg,
        "use_pca": use_pca,
        "val_accuracy": float(max(history.history["val_accuracy"])),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs_trained": len(history.history["loss"]),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[{region} | dropout={dropout} l2={l2_reg} pca={use_pca}] "
          f"val_acc={result['val_accuracy']:.4f} test_acc={result['test_accuracy']:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--region", required=True, choices=["XR_ELBOW", "XR_HAND", "XR_SHOULDER"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--use-pca", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-enhancement", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.index_csv)
    region_df = df[df["region"] == args.region]
    train_df = region_df[region_df["split"] == "train"].reset_index(drop=True)
    val_df = region_df[region_df["split"] == "val"].reset_index(drop=True)
    test_df = region_df[region_df["split"] == "test"].reset_index(drop=True)

    train_one_config(
        train_df, val_df, test_df, region=args.region,
        dropout=args.dropout, l2_reg=args.l2, use_pca=args.use_pca,
        output_dir=args.output_dir, batch_size=args.batch_size, epochs=args.epochs,
        apply_enhancement=not args.no_enhancement,
        pca_cache_path=os.path.join(args.output_dir, "pca.joblib") if args.use_pca else None,
    )


if __name__ == "__main__":
    main()
