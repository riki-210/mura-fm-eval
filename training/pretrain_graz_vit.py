"""
pretrain_graz_vit.py -- GRAZPEDWRI-DX domain-pretraining for the ViT-B16
backbone (ViT counterpart of pretrain_graz_backbone.py).

Two stages: (1) head only, backbone frozen, LR 1e-3; (2) full network, LR 1e-5
(ViTs need a lower LR than the CNN script's 1e-4). Saves ONLY the backbone as a
Keras 3 native .weights.h5 file, with a per-epoch checkpoint against walltime kills.
"""

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from vit_keras import vit

from preprocessing_v2 import build_tf_dataset_v2
from pretrain_graz_backbone import BackboneCheckpoint, ensure_weights_h5_suffix


def build_graz_vit_pretrain_model(img_size: int = 224, pretrained: bool = True):
    """ViT-B16 -> Dropout -> Dense(2). Returns (model, base)."""
    base = vit.vit_b16(image_size=img_size, include_top=False, pretrained=pretrained,
                       pretrained_top=False, weights="imagenet21k+imagenet2012")
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = vit.preprocess_inputs(inputs * 255.0)
    x = base(x, training=True)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation="softmax", name="graz_fracture_output")(x)
    model = models.Model(inputs, outputs, name="graz_pretrain_vit_b16")
    return model, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--output-weights", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15, help="max epochs for the full-backbone stage")
    ap.add_argument("--no-autocrop", action="store_true")
    ap.add_argument("--no-clahe", action="store_true")
    args = ap.parse_args()

    args.output_weights = ensure_weights_h5_suffix(args.output_weights)
    print(f"Backbone weights will be saved to: {args.output_weights}")

    df = pd.read_csv(args.index_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    train_ds = build_tf_dataset_v2(train_df, "has_fracture", args.batch_size, augment=True,
                                   use_autocrop=not args.no_autocrop, use_clahe=not args.no_clahe,
                                   shuffle=True, img_size=args.img_size)
    val_ds = build_tf_dataset_v2(val_df, "has_fracture", args.batch_size, augment=False,
                                 use_autocrop=not args.no_autocrop, use_clahe=not args.no_clahe,
                                 shuffle=False, img_size=args.img_size)

    train_labels = train_df["has_fracture"].values
    classes = np.unique(train_labels)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}
    print(f"GRAZPEDWRI-DX class distribution (train): {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    print(f"Class weights (balanced): {class_weight_dict}")

    model, base = build_graz_vit_pretrain_model(img_size=args.img_size)
    log_path = os.path.splitext(args.output_weights)[0] + "_pretrain_log.csv"

    print(f"\n===== Stage 1: training head only (ViT frozen), {args.warmup_epochs} epochs =====")
    base.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.warmup_epochs,
              class_weight=class_weight_dict,
              callbacks=[tf.keras.callbacks.CSVLogger(log_path, append=False)])

    print(f"\n===== Stage 2: full ViT fine-tuning on GRAZPEDWRI-DX, up to {args.epochs} epochs =====")
    base.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(log_path, append=True),
        BackboneCheckpoint(base, args.output_weights),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              class_weight=class_weight_dict, callbacks=callbacks)

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"GRAZPEDWRI-DX ViT pretraining done -- final val_acc={val_acc:.4f}")

    base.save_weights(args.output_weights)
    print(f"Saved backbone-only weights to {args.output_weights} "
          f"(use with train_tier2_vit.py --domain-weights {args.output_weights})")


if __name__ == "__main__":
    main()
