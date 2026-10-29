"""
train_tier2_strong.py -- accuracy-push Tier 2 training: a modern pretrained
backbone (DenseNet169 by default), auto-crop + CLAHE preprocessing, balanced
class weighting, and a two-phase fine-tuning schedule (freeze-then-unfreeze
with a lower fine-tuning LR), instead of AlexNet + single-phase training.

Two-phase fine-tuning rationale: training a randomly-initialized head on top
of a frozen pretrained backbone first (phase 1, higher LR) lets the head
reach a reasonable starting point before any gradient reaches the backbone;
only then is the whole network unfrozen and fine-tuned together at a much
lower LR (phase 2), which avoids the large, poorly-initialized head's
gradients from destroying the pretrained backbone's features early in
training -- a standard, well-evidenced transfer-learning recipe, and a
different (stronger) protocol than the single-phase fine-tuning used in
train_tier2_alexnet_pretrained.py.

Usage:
    python3 train_tier2_strong.py \
        --index-csv mura_ehr_index.csv --region XR_SHOULDER \
        --output-dir results/tier2_XR_SHOULDER_strong_densenet169 \
        --backbone DenseNet169 --dropout 0.5 --l2 1e-4
"""

import argparse
import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from preprocessing_v2 import build_tf_dataset_v2
from models_tier2_strong import build_strong_fracture_classifier, _BACKBONES


def predict_with_flip_tta(model, ds):
    """Average softmax over identity / h-flip / v-flip / h+v-flip of every batch.
    Returns (tta_probs, plain_probs). The training augmentation applies random
    left-right AND up-down flips, so all four views are in-distribution."""
    import tensorflow as tf
    fns = [lambda x: x,
           tf.image.flip_left_right,
           tf.image.flip_up_down,
           lambda x: tf.image.flip_up_down(tf.image.flip_left_right(x))]
    all_probs = []
    for fn in fns:
        view = ds.map(lambda x, y, fn=fn: (fn(x), y))
        all_probs.append(model.predict(view, verbose=0))
    return np.mean(all_probs, axis=0), all_probs[0]


def train_one_config(train_df, val_df, test_df, region: str, backbone: str,
                      dropout: float, l2_reg: float, output_dir: str,
                      img_size: int = 224, batch_size: int = 32,
                      warmup_epochs: int = 5, finetune_epochs: int = 60,
                      use_autocrop: bool = True, use_clahe: bool = True,
                      domain_weights_path: str = None, save_probs: bool = False,
                      tta: bool = False, seed: int = None) -> dict:
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

    # ---- Phase 1: frozen backbone, train only the new head ----
    print(f"\n===== Phase 1: training head only (backbone frozen), {warmup_epochs} epochs =====")
    model = build_strong_fracture_classifier(backbone, img_size=img_size, num_classes=2,
                                              dropout=dropout, l2_reg=l2_reg, freeze_backbone=True,
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

    # ---- Phase 2: unfreeze everything, fine-tune at a lower LR with cosine decay ----
    print(f"\n===== Phase 2: fine-tuning full network, up to {finetune_epochs} epochs =====")
    model2 = build_strong_fracture_classifier(backbone, img_size=img_size, num_classes=2,
                                               dropout=dropout, l2_reg=l2_reg, freeze_backbone=False,
                                               domain_weights_path=domain_weights_path)
    model2.load_weights(phase1_weights_path)

    steps_per_epoch = max(1, len(train_df) // batch_size)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5, decay_steps=steps_per_epoch * finetune_epochs,
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

    tta_info = {}
    if save_probs or tta:
        # Dumps predicted probabilities (+ true labels) for val and test, in
        # row order matching val_df/test_df -- consumed by
        # ensemble_stack_tier2.py to build a stacked meta-classifier across
        # several trained models for the same region, without needing to
        # reload every individual model at ensembling time.
        val_probs = model2.predict(val_ds)
        test_probs = model2.predict(test_ds)
        np.save(os.path.join(output_dir, "val_probs.npy"), val_probs)
        np.save(os.path.join(output_dir, "test_probs.npy"), test_probs)
        np.save(os.path.join(output_dir, "val_labels.npy"), val_df["fracture_label"].values)
        np.save(os.path.join(output_dir, "test_labels.npy"), test_df["fracture_label"].values)
        print(f"Saved val/test probabilities + labels to {output_dir} for ensembling.")

    if tta:
        val_tta, val_plain = predict_with_flip_tta(model2, val_ds)
        test_tta, test_plain = predict_with_flip_tta(model2, test_ds)
        np.save(os.path.join(output_dir, "val_probs_noaug.npy"), val_plain)
        np.save(os.path.join(output_dir, "test_probs_noaug.npy"), test_plain)
        np.save(os.path.join(output_dir, "val_probs.npy"), val_tta)      # ensembling/study-level scripts read these
        np.save(os.path.join(output_dir, "test_probs.npy"), test_tta)
        test_y = test_df["fracture_label"].values
        val_y = val_df["fracture_label"].values
        tta_info = {
            "tta": "flip4",
            "test_accuracy_no_tta": float((np.argmax(test_plain, 1) == test_y).mean()),
            "test_accuracy_tta": float((np.argmax(test_tta, 1) == test_y).mean()),
            "val_accuracy_no_tta": float((np.argmax(val_plain, 1) == val_y).mean()),
            "val_accuracy_tta": float((np.argmax(val_tta, 1) == val_y).mean()),
        }
        print(f"TTA (flip x4): test_acc {tta_info['test_accuracy_no_tta']:.4f} -> {tta_info['test_accuracy_tta']:.4f}")

    result = {
        "region": region,
        "backbone": backbone,
        "domain_weights_path": domain_weights_path,
        "dropout": dropout,
        "l2_reg": l2_reg,
        "use_autocrop": use_autocrop,
        "use_clahe": use_clahe,
        "warmup_epochs": warmup_epochs,
        "finetune_epochs_trained": len(history2.history["loss"]),
        "class_weight": class_weight_dict,
        "phase1_final_val_accuracy": float(history1.history["val_accuracy"][-1]),
        "val_accuracy": float(max(history2.history["val_accuracy"])),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "img_size": img_size,
        "seed": seed,
        **tta_info,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[{region} | STRONG backbone={backbone} dropout={dropout} l2={l2_reg} "
          f"autocrop={use_autocrop} clahe={use_clahe}] "
          f"val_acc={result['val_accuracy']:.4f} test_acc={result['test_accuracy']:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--region", required=True, choices=["XR_ELBOW", "XR_HAND", "XR_SHOULDER"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--backbone", default="DenseNet169", choices=list(_BACKBONES))
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--finetune-epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-autocrop", action="store_true")
    ap.add_argument("--no-clahe", action="store_true")
    ap.add_argument("--radimagenet-weights", "--domain-weights", dest="domain_weights", default=None,
                     help="Path to a domain-pretrained backbone .h5 weights file -- either a "
                          "RadImageNet release (ResNet50/DenseNet121/InceptionV3/IRV2 only) or a "
                          "GRAZPEDWRI-DX-pretrained file produced by pretrain_graz_backbone.py "
                          "(any backbone in _BACKBONES). If given, loads these weights into the "
                          "backbone instead of ImageNet weights.")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=None,
                     help="Seeds TF/numpy/python and the dataset shuffle order (default: leave unseeded as before).")
    ap.add_argument("--tta", action="store_true",
                     help="Flip-x4 test-time augmentation; saves TTA probs as val/test_probs.npy and plain ones as *_noaug.npy.")
    ap.add_argument("--save-probs", action="store_true",
                     help="Save val/test prediction probabilities + labels for later ensembling "
                          "via ensemble_stack_tier2.py.")
    args = ap.parse_args()

    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
        import data_utils
        data_utils.RNG_SEED = args.seed   # build_tf_dataset_v2 reads this at call time (shuffle order)
        print(f"Seed set to {args.seed}")

    df = pd.read_csv(args.index_csv)
    region_df = df[df["region"] == args.region]
    train_df = region_df[region_df["split"] == "train"].reset_index(drop=True)
    val_df = region_df[region_df["split"] == "val"].reset_index(drop=True)
    test_df = region_df[region_df["split"] == "test"].reset_index(drop=True)

    train_one_config(
        train_df, val_df, test_df, region=args.region, backbone=args.backbone,
        dropout=args.dropout, l2_reg=args.l2, output_dir=args.output_dir,
        batch_size=args.batch_size, warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        use_autocrop=not args.no_autocrop, use_clahe=not args.no_clahe,
        domain_weights_path=args.domain_weights, save_probs=args.save_probs,
        img_size=args.img_size, tta=args.tta, seed=args.seed,
    )


if __name__ == "__main__":
    main()
