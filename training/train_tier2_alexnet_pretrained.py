"""
train_tier2_alexnet_pretrained.py -- diagnostic variant of
train_tier2_alexnet.py that trains the PLAIN-IMAGE (non-PCA) AlexNet branch
with real ImageNet-pretrained weights instead of from-scratch initialization,
to test whether pretraining explains the gap between this reproduction's
Tier 2 accuracy (61-71% test) and the paper's claimed 92.71%.

Drop this file into the same codes/ folder as your existing scaffold
(alongside data_utils.py, models.py, pca_utils.py) -- it imports from
data_utils.py exactly like the original train_tier2_alexnet.py, and adds
models_alexnet_pretrained.py (also in this kit) as a new import.

One-time setup (see convert_pretrained_alexnet_weights.py's docstring):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    python3 convert_pretrained_alexnet_weights.py --out alexnet_imagenet_weights.npz

Usage (fine-tune the whole pretrained network, paper's best dropout/L2):
    python3 train_tier2_alexnet_pretrained.py \
        --index-csv mura_ehr_index.csv --region XR_SHOULDER \
        --output-dir results/tier2_XR_SHOULDER_pretrained \
        --pretrained-weights alexnet_imagenet_weights.npz \
        --dropout 0.8 --l2 1e-4

Usage (freeze the backbone, train only the new head -- faster first check):
    ... same as above, plus --freeze-backbone
"""

import argparse
import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from data_utils import build_tf_dataset, IMG_SIZE
from models_alexnet_pretrained import build_alexnet_pretrained


def train_one_config(train_df, val_df, test_df, region: str, dropout: float, l2_reg: float,
                      pretrained_weights_path: str, freeze_backbone: bool, output_dir: str,
                      img_size: int = IMG_SIZE, batch_size: int = 32, epochs: int = 100,
                      apply_enhancement: bool = True) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    train_ds = build_tf_dataset(train_df, "fracture_label", batch_size, augment=True,
                                 apply_enhancement=apply_enhancement, shuffle=True, img_size=img_size)
    val_ds = build_tf_dataset(val_df, "fracture_label", batch_size, augment=False,
                               apply_enhancement=apply_enhancement, shuffle=False, img_size=img_size)
    test_ds = build_tf_dataset(test_df, "fracture_label", batch_size, augment=False,
                                apply_enhancement=apply_enhancement, shuffle=False, img_size=img_size)

    # Balanced class weighting -- added after discovering XR_HAND collapsed
    # to predicting the majority class every epoch (val_accuracy frozen
    # bit-for-bit at 0.7578475475311279 across all 6 epochs with no
    # weighting). No Tier 2 script originally passed class_weight to
    # model.fit(), unlike Paper 1's pipeline, which had to fix an almost
    # identical collapse with inverse-frequency weighting.
    train_labels = train_df["fracture_label"].values
    classes = np.unique(train_labels)
    class_weight_values = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weight_values)}
    print(f"Class distribution (train): {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    print(f"Class weights (balanced): {class_weight_dict}")

    model = build_alexnet_pretrained(
        weights_npz_path=pretrained_weights_path, img_size=img_size, num_classes=2,
        dropout=dropout, l2_reg=l2_reg, freeze_backbone=freeze_backbone,
    )

    # Fine-tuning a pretrained backbone usually wants a smaller LR than
    # training from scratch, to avoid destroying the pretrained features in
    # the first few steps; 1e-5 here vs. 1e-4 for the from-scratch model in
    # train_tier2_alexnet.py. When freeze_backbone=True only the new head's
    # (randomly initialized) weights are updated, so 1e-4 is fine either way.
    lr = 1e-4 if freeze_backbone else 1e-5
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
        "pretrained": True,
        "freeze_backbone": freeze_backbone,
        "class_weight": class_weight_dict,
        "val_accuracy": float(max(history.history["val_accuracy"])),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs_trained": len(history.history["loss"]),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[{region} | PRETRAINED dropout={dropout} l2={l2_reg} freeze={freeze_backbone}] "
          f"val_acc={result['val_accuracy']:.4f} test_acc={result['test_accuracy']:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--region", required=True, choices=["XR_ELBOW", "XR_HAND", "XR_SHOULDER"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pretrained-weights", required=True,
                     help="Path to the .npz produced by convert_pretrained_alexnet_weights.py")
    ap.add_argument("--dropout", type=float, default=0.8)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--freeze-backbone", action="store_true")
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
        freeze_backbone=args.freeze_backbone,
        output_dir=args.output_dir, batch_size=args.batch_size, epochs=args.epochs,
        apply_enhancement=not args.no_enhancement,
    )


if __name__ == "__main__":
    main()
