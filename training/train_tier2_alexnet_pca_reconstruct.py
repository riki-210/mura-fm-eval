"""
train_tier2_alexnet_pca_reconstruct.py -- tests the third PCA interpretation
(see pca_reconstruct_utils.py's module docstring): PCA as an image-space
DENOISING/RECONSTRUCTION step feeding the FULL AlexNet (conv stack intact,
~60M params, matching the paper's Table 4), rather than a PCA feature
vector feeding a truncated FC-only head (the original pca_utils.py reading).

Runs with --pretrained-weights (recommended, since the plain-pretrained
diagnostic already showed pretraining alone only buys ~3pts -- the next
question is whether PCA-conditioning input helps MORE when the backbone can
actually exploit it) or without it (from-scratch AlexNet + PCA-reconstructed
input, closer to a literal reading of "AlexNet (from the original 2012
architecture) + PCA" with no transfer learning at all). Both are worth
trying; running with --pretrained-weights first is the single best next
experiment given everything found so far.

Drop this file (and pca_reconstruct_utils.py) into the same codes/ folder as
your existing scaffold. Needs models.py (build_alexnet) for the from-scratch
path and models_alexnet_pretrained.py (build_alexnet_pretrained, from the
earlier addon kit) for the --pretrained-weights path -- both should already
be in codes/ from the previous round.

Usage (pretrained backbone + PCA-reconstructed input, paper's best config):
    python3 train_tier2_alexnet_pca_reconstruct.py \
        --index-csv mura_ehr_index.csv --region XR_SHOULDER \
        --output-dir results/tier2_XR_SHOULDER_pca_reconstruct_pretrained \
        --pretrained-weights alexnet_imagenet_weights.npz \
        --dropout 0.8 --l2 1e-4

Usage (from-scratch AlexNet + PCA-reconstructed input):
    python3 train_tier2_alexnet_pca_reconstruct.py \
        --index-csv mura_ehr_index.csv --region XR_SHOULDER \
        --output-dir results/tier2_XR_SHOULDER_pca_reconstruct_scratch \
        --dropout 0.8 --l2 1e-4
"""

import argparse
import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from data_utils import IMG_SIZE
from pca_reconstruct_utils import fit_pca, reconstruct_image, save_pca, load_pca


def build_reconstructed_datasets(train_df, val_df, test_df, pca, apply_enhancement,
                                  batch_size, img_size):
    """
    Pre-computes the PCA-reconstructed (still image-shaped) input for every
    image up front -- a one-off CPU pass (PCA transform/inverse_transform
    are sklearn, not TF-graph-compatible), same pattern as the original
    pca_utils.py's build_pca_datasets. Returns tf.data.Dataset objects
    yielding (image, label) batches, image shape (img_size, img_size, 3).

    Memory note: holding one region's full reconstructed train set as a
    float32 array (img_size x img_size x 3 x N) is a few GB for the largest
    region (shoulder, ~6.3k train images -> ~3.8GB) -- comfortably inside a
    32GB job, and only one region is held in memory at a time since this
    script processes one region per invocation (looped externally by the
    .pbs job, process exits and frees memory between regions).
    """
    def featurize(df):
        X = np.stack([
            reconstruct_image(pca, p, output_size=img_size, apply_enhancement=apply_enhancement)
            for p in df["image_path"]
        ])
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
                      output_dir: str, pretrained_weights_path: str = None,
                      img_size: int = IMG_SIZE, batch_size: int = 32, epochs: int = 100,
                      apply_enhancement: bool = True, pca_cache_path: str = None) -> dict:
    os.makedirs(output_dir, exist_ok=True)

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

    train_ds, val_ds, test_ds = build_reconstructed_datasets(
        train_df, val_df, test_df, pca, apply_enhancement, batch_size, img_size
    )

    # Balanced class weighting -- added after discovering that XR_HAND
    # collapsed to predicting the majority class every epoch (val_accuracy
    # frozen bit-for-bit at 0.7578475475311279 across all 6 epochs in the
    # plain-pretrained diagnostic run). None of the original Tier 2 scripts
    # passed class_weight to model.fit(), unlike Paper 1's pipeline, which
    # already had to fix an almost identical collapse via inverse-frequency
    # weighting. Computed fresh per-region/per-config from the actual
    # training split, not hardcoded, since the three regions' imbalance
    # likely differs.
    train_labels = train_df["fracture_label"].values
    classes = np.unique(train_labels)
    class_weight_values = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weight_values)}
    print(f"Class distribution (train): {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    print(f"Class weights (balanced): {class_weight_dict}")

    if pretrained_weights_path:
        from models_alexnet_pretrained import build_alexnet_pretrained
        model = build_alexnet_pretrained(
            weights_npz_path=pretrained_weights_path, img_size=img_size, num_classes=2,
            dropout=dropout, l2_reg=l2_reg, freeze_backbone=False,
        )
        lr = 1e-5  # fine-tuning a pretrained backbone -- same rationale as train_tier2_alexnet_pretrained.py
    else:
        from models import build_alexnet
        model = build_alexnet(img_size=img_size, num_classes=2, dropout=dropout, l2_reg=l2_reg,
                               input_channels=3)
        lr = 1e-4  # from-scratch -- matches train_tier2_alexnet.py's plain-image branch

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_path = os.path.join(output_dir, "best_model.h5")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.CSVLogger(os.path.join(output_dir, "training_log.csv")),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks,
                         batch_size=batch_size, class_weight=class_weight_dict)
    test_loss, test_acc = model.evaluate(test_ds)

    result = {
        "region": region,
        "dropout": dropout,
        "l2_reg": l2_reg,
        "pca_reconstruction": True,
        "pretrained": bool(pretrained_weights_path),
        "val_accuracy": float(max(history.history["val_accuracy"])),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs_trained": len(history.history["loss"]),
        "pca_n_components": int(pca.n_components_),
        "class_weight": class_weight_dict,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[{region} | PCA-RECONSTRUCT dropout={dropout} l2={l2_reg} "
          f"pretrained={bool(pretrained_weights_path)}] "
          f"val_acc={result['val_accuracy']:.4f} test_acc={result['test_accuracy']:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--region", required=True, choices=["XR_ELBOW", "XR_HAND", "XR_SHOULDER"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pretrained-weights", default=None,
                     help="Optional path to the .npz from convert_pretrained_alexnet_weights.py. "
                          "If omitted, trains AlexNet from scratch on the PCA-reconstructed input.")
    ap.add_argument("--dropout", type=float, default=0.8)
    ap.add_argument("--l2", type=float, default=1e-4)
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
        dropout=args.dropout, l2_reg=args.l2,
        pretrained_weights_path=args.pretrained_weights,
        output_dir=args.output_dir, batch_size=args.batch_size, epochs=args.epochs,
        apply_enhancement=not args.no_enhancement,
        pca_cache_path=os.path.join(args.output_dir, "pca.joblib"),
    )


if __name__ == "__main__":
    main()
