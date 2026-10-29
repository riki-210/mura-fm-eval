"""
pretrain_graz_backbone.py -- domain-pretrains a Tier-2 backbone on
GRAZPEDWRI-DX (pediatric wrist radiographs, fracture-vs-no-fracture) BEFORE
fine-tuning on MURA Tier 2, instead of starting Tier 2 training from
ImageNet weights directly.

Why this is worth trying (the user's own suggestion): ImageNet pretraining
transfers generic natural-image features (edges, textures, object shapes)
to a bone-radiograph task across a large domain gap. RadImageNet closes part
of that gap (radiology images in general). GRAZPEDWRI-DX closes it further
still for the specific sub-problem of "does this bone X-ray show a
fracture" -- it IS a bone-fracture-detection X-ray dataset, just a
different body region (pediatric wrist, not MURA's elbow/hand/shoulder) and
different patient population (pediatric, open growth plates, vs. MURA's
mixed-age-skewed-adult). That is a genuine domain shift, not a data leak:

  * GRAZPEDWRI-DX and MURA have ZERO patient, region, or label overlap --
    GRAZPEDWRI-DX is wrist-only pediatric trauma radiographs with YOLO
    bounding-box fracture annotations; MURA is elbow/hand/shoulder (no
    wrist) radiographs, mostly adult, with study-level fracture labels and
    no boxes. Pretraining on one and fine-tuning on the other is standard
    transfer learning, not evaluating on the pretraining data.
  * This is therefore NOT "combining the two datasets' labels into one
    bigger training set" (which the user asked about and which this project
    correctly does NOT do, since GRAZPEDWRI-DX's pediatric wrist cases are
    not valid additional training examples for MURA's elbow/hand/shoulder
    fracture task -- different body parts, different bone biology/growth
    plates, would corrupt the region-specific classifiers). It IS valid,
    standard sequential transfer learning: learn generic
    "what-does-a-fractured-bone-look-like-on-X-ray" features on
    GRAZPEDWRI-DX first, then specialize to each MURA region.

What this script does:
  1. Loads a GRAZPEDWRI-DX image index CSV (image_path, has_fracture columns
     -- see "Preparing the GRAZPEDWRI-DX index" below for how to build this
     from the released YOLO-format annotations).
  2. Builds <backbone> -> GlobalAveragePooling -> Dropout -> Dense(2,softmax)
     (same shape as models_tier2_strong.build_strong_fracture_classifier,
     via the SAME _BACKBONES registry, so the resulting weights file matches
     that function's backbone instantiation exactly).
  3. Trains the whole network (backbone unfrozen from the start -- this is a
     from-ImageNet single-stage pretraining run, not the two-phase Tier 2
     recipe, since the goal here is just to adapt the backbone's features to
     bone-radiograph imagery as well as possible, not to produce a final
     fracture classifier) on GRAZPEDWRI-DX's own train/val split.
  4. Saves ONLY the backbone's weights (not the GRAZPEDWRI-DX-specific head)
     to an .h5 file, in a layer-name-addressable format suitable for
     `base.load_weights(path, by_name=True, skip_mismatch=True)` --
     train_tier2_strong.py's `--domain-weights` flag loads this file exactly
     the same way it loads a RadImageNet file.

Preparing the GRAZPEDWRI-DX index CSV:
    GRAZPEDWRI-DX ships YOLO-format .txt annotation files (one per image,
    one line per bounding box, empty file = no fracture visible). Build the
    index with a short script that, for each image, checks whether its
    paired .txt annotation file is non-empty (has_fracture=1) or empty/
    missing (has_fracture=0), e.g.:

        import os, pandas as pd
        rows = []
        for img_name in os.listdir(IMAGES_DIR):
            stem, _ = os.path.splitext(img_name)
            label_path = os.path.join(LABELS_DIR, stem + ".txt")
            has_fracture = int(os.path.exists(label_path) and os.path.getsize(label_path) > 0)
            rows.append({"image_path": os.path.join(IMAGES_DIR, img_name), "has_fracture": has_fracture})
        df = pd.DataFrame(rows)
        # simple deterministic 80/10/10 split, seeded for reproducibility
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df); n_train = int(0.8*n); n_val = int(0.1*n)
        df["split"] = ["train"]*n_train + ["val"]*n_val + ["test"]*(n - n_train - n_val)
        df.to_csv("graz_index.csv", index=False)

    GRAZPEDWRI-DX's released metadata.csv may already carry fracture-present
    flags per image (check the dataset's own documentation) -- use that
    directly instead of deriving it from box files if available; the
    derivation above is a fallback that works from the raw YOLO labels alone.

Usage:
    python3 pretrain_graz_backbone.py \
        --index-csv graz_index.csv \
        --output-weights graz_pretrained_densenet169.h5 \
        --backbone DenseNet169 --epochs 30
"""

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models

from preprocessing_v2 import build_tf_dataset_v2
from models_tier2_strong import _BACKBONES


def ensure_weights_h5_suffix(path: str) -> str:
    """
    TensorFlow 2.16's bundled Keras 3 enforces that any `save_weights()` /
    `load_weights()` target end in the literal suffix `.weights.h5` (its
    native weights-only format) -- a plain `.h5` (which was accepted by the
    Keras 2 / older-TF `save_weights` API this project was originally
    written against) now raises `ValueError: The filename must end in
    `.weights.h5``. Normalizing every weights-only path through this
    function means a script works regardless of what extension the caller
    (a human typing a --output-weights path, or an older PBS script) passes.
    """
    if path.endswith(".weights.h5"):
        return path
    if path.endswith(".h5"):
        return path[: -len(".h5")] + ".weights.h5"
    return path + ".weights.h5"


class BackboneCheckpoint(tf.keras.callbacks.Callback):
    """
    Saves ONLY the backbone's weights (not the GRAZPEDWRI-DX-specific head)
    to `output_path` at the end of every epoch, overwriting the same file
    each time. This is a safety net against a PBS walltime kill: without it,
    the backbone is only saved once, after `model.fit()` returns -- if the
    job is killed mid-training (walltime exceeded, node failure, etc.), there
    would be NO usable output at all, even though many epochs of real GPU
    training already happened. With this callback, the most recent epoch's
    backbone weights are always on disk and usable with
    train_tier2_strong.py's --domain-weights flag, even if the run never
    reaches its final epoch.
    """
    def __init__(self, base_model: tf.keras.Model, output_path: str):
        super().__init__()
        self.base_model = base_model
        self.output_path = output_path

    def on_epoch_end(self, epoch, logs=None):
        self.base_model.save_weights(self.output_path)
        print(f"[BackboneCheckpoint] Saved backbone weights after epoch {epoch + 1} to {self.output_path}")


def build_graz_pretrain_model(backbone_name: str, img_size: int = 224):
    if backbone_name not in _BACKBONES:
        raise ValueError(f"Unknown backbone {backbone_name!r}; choices: {list(_BACKBONES)}")
    backbone_cls, preprocess_fn = _BACKBONES[backbone_name]

    base = backbone_cls(include_top=False, weights="imagenet",
                         input_shape=(img_size, img_size, 3), pooling="avg")

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = preprocess_fn(inputs * 255.0)
    x = base(x, training=True)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation="softmax", name="graz_fracture_output")(x)
    model = models.Model(inputs, outputs, name=f"graz_pretrain_{backbone_name}")
    return model, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True,
                     help="GRAZPEDWRI-DX index CSV with columns: image_path, has_fracture, split "
                          "(split in {train,val,test}) -- see this script's docstring for how to build it.")
    ap.add_argument("--output-weights", required=True,
                     help="Where to save the pretrained backbone-only weights (.h5), for later use "
                          "with train_tier2_strong.py's --domain-weights flag.")
    ap.add_argument("--backbone", default="DenseNet169", choices=list(_BACKBONES))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
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
    class_weight_values = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weight_values)}
    print(f"GRAZPEDWRI-DX class distribution (train): "
          f"{dict(zip(*np.unique(train_labels, return_counts=True)))}")
    print(f"Class weights (balanced): {class_weight_dict}")

    model, base = build_graz_pretrain_model(args.backbone, img_size=args.img_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(os.path.splitext(args.output_weights)[0] + "_pretrain_log.csv"),
        # Safety net against a PBS walltime kill -- see BackboneCheckpoint's
        # docstring. Without this, a job killed mid-training leaves nothing
        # usable at all, no matter how many epochs of real training happened.
        BackboneCheckpoint(base, args.output_weights),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              class_weight=class_weight_dict, callbacks=callbacks)

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"GRAZPEDWRI-DX pretraining done -- backbone={args.backbone} final val_acc={val_acc:.4f}")

    # Final save, after EarlyStopping's restore_best_weights has put the
    # BEST epoch's weights back into `model` (and therefore into `base`,
    # which is a sub-model of it) -- overwrites the per-epoch checkpoint
    # above with the best-epoch version specifically, in the normal case
    # where training completes without being killed.
    base.save_weights(args.output_weights)
    print(f"Saved backbone-only weights to {args.output_weights} "
          f"(use with train_tier2_strong.py --domain-weights {args.output_weights} "
          f"--backbone {args.backbone})")


if __name__ == "__main__":
    main()
