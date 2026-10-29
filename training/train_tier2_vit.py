"""
train_tier2_vit.py -- Vision Transformer (ViT-B16) Tier 2 fracture
classifier training, mirroring train_tier2_strong.py's two-phase
fine-tuning recipe (frozen-backbone warmup -> full fine-tune with cosine
LR decay), same preprocessing (autocrop + CLAHE) and balanced class
weighting, so results are directly comparable in the final results table.

Requires: pip install vit-keras (see models_tier2_vit.py's docstring).

Usage:
    python3 train_tier2_vit.py \
        --index-csv mura_ehr_index.csv --region XR_SHOULDER \
        --output-dir results/tier2_XR_SHOULDER_vit_b16 \
        --dropout 0.5 --l2 1e-4 --save-probs
"""

import argparse
import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from preprocessing_v2 import build_tf_dataset_v2
from models_tier2_vit import build_vit_fracture_classifier


def train_one_config(train_df, val_df, test_df, region: str, vit_variant: str,
                      dropout: float, l2_reg: float, output_dir: str,
                      img_size: int = 224, batch_size: int = 32,
                      warmup_epochs: int = 5, finetune_epochs: int = 60,
                      use_autocrop: bool = True, use_clahe: bool = True,
                      save_probs: bool = False, domain_weights_path: str = None) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    train_ds = build_tf_dataset_v2(train_df, "fracture_label", batch_size, augment=True,
                                    use_autocrop=use_autocrop, use_clahe=use_clahe,
                                    shuffle=True, img_size=img_size)
    val_ds = build_tf_dataset_v2(val_df, "fracture_label", batch_size, augment=False,
                                  use_autocrop=use_autocrop, use_clahe=use_clahe,
                                  shuffle=False, img_size=img_size)
    test_ds = build_tf_dataset_v2(test_df, "fracture_label", batch_size, augment=False,
                                   use_autocrop=use_autocrop, use_clahe=use_clahe,
                                   shuffle=False, img_size=img_size)

    train_labels = train_df["fracture_label"].values
    classes = np.unique(train_labels)
    class_weight_values = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weight_values)}
    print(f"Class distribution (train): {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    print(f"Class weights (balanced): {class_weight_dict}")

    csv_log_path = os.path.join(output_dir, "training_log.csv")

    print(f"\n===== Phase 1: training head only (ViT frozen), {warmup_epochs} epochs =====")
    model = build_vit_fracture_classifier(img_size=img_size, num_classes=2, dropout=dropout,
                                           l2_reg=l2_reg, freeze_backbone=True, vit_variant=vit_variant,
                                           domain_weights_path=domain_weights_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=warmup_epochs,
        class_weight=class_weight_dict,
        callbacks=[tf.keras.callbacks.CSVLogger(csv_log_path, append=False)],
    )

    phase1_weights_path = os.path.join(output_dir, "phase1_head_weights.weights.h5")
    model.save_weights(phase1_weights_path)

    print(f"\n===== Phase 2: fine-tuning full ViT, up to {finetune_epochs} epochs =====")
    model2 = build_vit_fracture_classifier(img_size=img_size, num_classes=2, dropout=dropout,
                                            l2_reg=l2_reg, freeze_backbone=False, vit_variant=vit_variant,
                                            domain_weights_path=domain_weights_path)
    model2.load_weights(phase1_weights_path)

    steps_per_epoch = max(1, len(train_df) // batch_size)
    # ViTs are more prone to destabilizing during full fine-tuning than CNNs
    # -- an even lower peak LR than the CNN backbones' 1e-5 is standard
    # practice for ViT fine-tuning (commonly 1e-5 to 5e-6 range).
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-6, decay_steps=steps_per_epoch * finetune_epochs,
    )
    model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    ckpt_path = os.path.join(output_dir, "best_model.h5")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.CSVLogger(csv_log_path, append=True),
    ]
    history2 = model2.fit(
        train_ds, validation_data=val_ds, epochs=finetune_epochs,
        class_weight=class_weight_dict, callbacks=callbacks,
    )

    test_loss, test_acc = model2.evaluate(test_ds)

    if save_probs:
        val_probs = model2.predict(val_ds)
        test_probs = model2.predict(test_ds)
        np.save(os.path.join(output_dir, "val_probs.npy"), val_probs)
        np.save(os.path.join(output_dir, "test_probs.npy"), test_probs)
        np.save(os.path.join(output_dir, "val_labels.npy"), val_df["fracture_label"].values)
        np.save(os.path.join(output_dir, "test_labels.npy"), test_df["fracture_label"].values)
        print(f"Saved val/test probabilities + labels to {output_dir} for ensembling.")

    result = {
        "region": region,
        "model": f"ViT-{vit_variant}",
        "domain_weights_path": domain_weights_path,
        "dropout": dropout,
        "l2_reg": l2_reg,
        "use_autocrop": use_autocrop,
        "use_clahe": use_clahe,
        "class_weight": class_weight_dict,
        "phase1_final_val_accuracy": float(history1.history["val_accuracy"][-1]),
        "val_accuracy": float(max(history2.history["val_accuracy"])),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[{region} | ViT-{vit_variant} dropout={dropout} l2={l2_reg}] "
          f"val_acc={result['val_accuracy']:.4f} test_acc={result['test_accuracy']:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--region", required=True, choices=["XR_ELBOW", "XR_HAND", "XR_SHOULDER"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--vit-variant", default="vit_b16", choices=["vit_b16", "vit_b32", "vit_l16", "vit_l32"])
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--finetune-epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)  # ViT-B16 is heavier than AlexNet/DenseNet -- smaller default batch
    ap.add_argument("--no-autocrop", action="store_true")
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--save-probs", action="store_true")
    ap.add_argument("--domain-weights", default=None,
                     help="GRAZPEDWRI-DX-pretrained ViT backbone (.weights.h5) from pretrain_graz_vit.py.")
    args = ap.parse_args()

    df = pd.read_csv(args.index_csv)
    region_df = df[df["region"] == args.region]
    train_df = region_df[region_df["split"] == "train"].reset_index(drop=True)
    val_df = region_df[region_df["split"] == "val"].reset_index(drop=True)
    test_df = region_df[region_df["split"] == "test"].reset_index(drop=True)

    train_one_config(
        train_df, val_df, test_df, region=args.region, vit_variant=args.vit_variant,
        dropout=args.dropout, l2_reg=args.l2, output_dir=args.output_dir,
        batch_size=args.batch_size, warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        use_autocrop=not args.no_autocrop, use_clahe=not args.no_clahe,
        save_probs=args.save_probs, domain_weights_path=args.domain_weights,
    )


if __name__ == "__main__":
    main()
