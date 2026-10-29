"""
data_utils.py -- MURA dataset indexing, subject-level splitting, and TF data
pipelines for Paper 2 (MediInsight), restricted to the three regions used in
the paper: elbow, hand, shoulder.

MURA's official release only ships train/ and valid/ splits (no held-out
test set). The paper reports a 70/15/15 train/val/test split, so this module
re-splits the *combined* official train+valid pool into a fresh 70/15/15
split, at the PATIENT level (never splitting one patient's studies across
partitions) -- exactly the same subject-level discipline used in Paper 1's
ADNI pipeline, and for the same reason: MURA patients often contribute
several studies, and mixing one patient's studies across train/val/test
would leak information and inflate reported accuracy.

Expected on-disk layout (standard MURA-v1.1):
    <mura_root>/train/XR_ELBOW/patientNNNNN/studyN_positive|negative/*.png
    <mura_root>/train/XR_HAND/...
    <mura_root>/train/XR_SHOULDER/...
    <mura_root>/valid/XR_ELBOW/...
    <mura_root>/valid/XR_HAND/...
    <mura_root>/valid/XR_SHOULDER/...

XR_FINGER, XR_FOREARM, XR_HUMERUS, XR_WRIST are present on disk (since the
full MURA-v1.1 archive was downloaded) but are deliberately excluded here --
the paper's reported numbers are for elbow/hand/shoulder only.
"""

import os
import re
import glob
import numpy as np
import pandas as pd

REGIONS = ["XR_ELBOW", "XR_HAND", "XR_SHOULDER"]
REGION_LABEL_MAP = {r: i for i, r in enumerate(REGIONS)}  # for Tier 1 (region ID)

IMG_SIZE = 224  # matches the paper's 224x224 3-channel input spec

RNG_SEED = 42


def _label_from_study_dirname(study_dirname: str) -> int:
    """study1_positive -> 1 (fracture present); study1_negative -> 0."""
    if "positive" in study_dirname:
        return 1
    if "negative" in study_dirname:
        return 0
    raise ValueError(f"Cannot parse label from study dir name: {study_dirname}")


def index_mura(mura_root: str, regions=REGIONS) -> pd.DataFrame:
    """
    Walks <mura_root>/{train,valid}/<region>/patientNNNNN/studyN_label/*.png
    and returns one row per image with columns:
        image_path, patient_id, region, region_label, fracture_label, orig_split
    orig_split is MURA's own train/valid tag; it is NOT used as the final
    split (see make_patient_level_split below) but is kept for reference.
    """
    rows = []
    for orig_split in ("train", "valid"):
        for region in regions:
            region_dir = os.path.join(mura_root, orig_split, region)
            if not os.path.isdir(region_dir):
                raise FileNotFoundError(
                    f"Expected region folder not found: {region_dir}\n"
                    f"Check --mura-root points at the folder containing train/ and valid/."
                )
            patient_dirs = sorted(glob.glob(os.path.join(region_dir, "patient*")))
            for pdir in patient_dirs:
                patient_id = os.path.basename(pdir)
                study_dirs = sorted(glob.glob(os.path.join(pdir, "study*")))
                for sdir in study_dirs:
                    label = _label_from_study_dirname(os.path.basename(sdir))
                    for img_path in sorted(glob.glob(os.path.join(sdir, "*.png"))):
                        rows.append({
                            "image_path": img_path,
                            "patient_id": patient_id,
                            "region": region,
                            "region_label": REGION_LABEL_MAP[region],
                            "fracture_label": label,
                            "orig_split": orig_split,
                        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f"No images found under {mura_root} for regions {regions}.")
    return df


def make_patient_level_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15,
                              seed=RNG_SEED) -> pd.DataFrame:
    """
    Adds a 'split' column (train/val/test) to df, computed independently
    within each region so the 70/15/15 ratio holds per-region (matching the
    paper's per-region reporting), splitting at the patient level so no
    patient's images appear in more than one split.
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["split"] = ""

    for region in df["region"].unique():
        region_mask = df["region"] == region
        patients = np.array(df.loc[region_mask, "patient_id"].unique().tolist())
        rng.shuffle(patients)

        n = len(patients)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train_patients = set(patients[:n_train])
        val_patients = set(patients[n_train:n_train + n_val])
        test_patients = set(patients[n_train + n_val:])

        def assign(pid):
            if pid in train_patients:
                return "train"
            if pid in val_patients:
                return "val"
            return "test"

        df.loc[region_mask, "split"] = df.loc[region_mask, "patient_id"].apply(assign)

    return df


def print_split_summary(df: pd.DataFrame):
    print("\n=== MURA (elbow/hand/shoulder) patient-level split summary ===")
    for region in REGIONS:
        sub = df[df["region"] == region]
        print(f"\n{region}:")
        for split in ("train", "val", "test"):
            s = sub[sub["split"] == split]
            n_patients = s["patient_id"].nunique()
            n_images = len(s)
            n_pos = (s["fracture_label"] == 1).sum()
            n_neg = (s["fracture_label"] == 0).sum()
            print(f"  {split:5s}: {n_images:5d} images, {n_patients:4d} patients "
                  f"({n_pos} positive / {n_neg} negative)")
    print(f"\nTotal images (elbow+hand+shoulder): {len(df)}")
    print(f"Total unique patients: {df['patient_id'].nunique()}")


# ---------------------------------------------------------------------------
# Image loading (enhancement pipeline applied here, at load time)
# ---------------------------------------------------------------------------
def load_and_enhance(image_path: str, apply_enhancement: bool = True,
                      img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Loads one MURA PNG, resizes to img_size x img_size, applies the four-stage
    enhancement pipeline (enhancement.py) if apply_enhancement, and returns a
    (img_size, img_size, 3) float32 array in [0, 1] (3-channel, per the
    paper's spec, by replicating the single enhanced channel).
    """
    import cv2
    from enhancement import enhance_image

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    if apply_enhancement:
        enhanced = enhance_image(img)  # float32 [0,1], single channel
    else:
        enhanced = img.astype(np.float32) / 255.0

    rgb = np.stack([enhanced, enhanced, enhanced], axis=-1)
    return rgb.astype(np.float32)


def build_tf_dataset(df: pd.DataFrame, label_col: str, batch_size: int = 32,
                      augment: bool = False, apply_enhancement: bool = True,
                      shuffle: bool = False, img_size: int = IMG_SIZE):
    """
    Builds a tf.data.Dataset yielding (image, label) batches for the given
    dataframe slice (already filtered to one region/split by the caller).

    augment=True applies the paper's train-only augmentation: horizontal/
    vertical flips, +/-30 deg rotation, +/-20% zoom, +/-10% brightness/contrast
    jitter, elastic deformation -- implemented via
    tf.keras.layers augmentation layers plus a custom elastic-deformation
    map function (TF has no built-in elastic deform).
    """
    import tensorflow as tf

    paths = df["image_path"].values
    labels = df[label_col].values.astype(np.int64)

    def _load(path_tensor, label_tensor):
        path = path_tensor.numpy().decode("utf-8")
        img = load_and_enhance(path, apply_enhancement=apply_enhancement, img_size=img_size)
        return img, label_tensor

    def _tf_load(path_tensor, label_tensor):
        img, label = tf.py_function(_load, [path_tensor, label_tensor], [tf.float32, tf.int64])
        img.set_shape([img_size, img_size, 3])
        label.set_shape([])
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=RNG_SEED, reshuffle_each_iteration=True)
    ds = ds.map(_tf_load, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(lambda x, y: (_augment_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _augment_image(img):
    """Train-only augmentation, applied per the paper's spec, via tf ops."""
    import tensorflow as tf

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    # +/-30 degree rotation
    angle = tf.random.uniform([], minval=-30.0, maxval=30.0) * (np.pi / 180.0)
    img = _rotate_image(img, angle)

    # +/-20% zoom (random crop + resize back)
    img = _random_zoom(img, min_zoom=0.8, max_zoom=1.2)

    # +/-10% brightness/contrast jitter
    img = tf.image.random_brightness(img, max_delta=0.10)
    img = tf.image.random_contrast(img, lower=0.90, upper=1.10)
    img = tf.clip_by_value(img, 0.0, 1.0)

    # Elastic deformation (lightweight approximation)
    img = _elastic_deform(img)
    return img


def _rotate_image(img, angle_rad):
    import tensorflow as tf
    try:
        import tensorflow_addons as tfa
        return tfa.image.rotate(img, angle_rad, interpolation="bilinear")
    except ImportError:
        # tensorflow_addons not installed -- fall back to 90-degree-multiple
        # rotations only (coarser than the paper's continuous +/-30deg, but
        # keeps the pipeline runnable without an extra dependency). Install
        # tensorflow_addons for the full continuous-angle behaviour.
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        return tf.image.rot90(img, k=k)


def _random_zoom(img, min_zoom=0.8, max_zoom=1.2):
    import tensorflow as tf
    shape = tf.shape(img)
    h, w = shape[0], shape[1]
    scale = tf.random.uniform([], min_zoom, max_zoom)
    new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    resized = tf.image.resize(img, [new_h, new_w])
    resized = tf.image.resize_with_crop_or_pad(resized, h, w)
    return resized


def _elastic_deform(img, alpha=8.0, sigma=3.0):
    """
    Lightweight elastic deformation via a smoothed random displacement field,
    approximated with tf ops (no scipy dependency inside the tf.data graph).
    This is a simplification of classical elastic deformation (Simard et al.)
    -- adequate as a training-time regularizer, not a bit-exact reproduction
    of any specific reference implementation.
    """
    import tensorflow as tf
    shape = tf.shape(img)
    h, w = shape[0], shape[1]

    dx = tf.random.normal([h, w], mean=0.0, stddev=alpha)
    dy = tf.random.normal([h, w], mean=0.0, stddev=alpha)
    # Smooth the displacement field with a small average-pool blur.
    dx = tf.nn.avg_pool2d(dx[None, :, :, None], ksize=9, strides=1, padding="SAME")[0, :, :, 0]
    dy = tf.nn.avg_pool2d(dy[None, :, :, None], ksize=9, strides=1, padding="SAME")[0, :, :, 0]

    xs, ys = tf.meshgrid(tf.range(w), tf.range(h))
    xs = tf.cast(xs, tf.float32) + dx
    ys = tf.cast(ys, tf.float32) + dy
    xs = tf.clip_by_value(xs, 0.0, tf.cast(w - 1, tf.float32))
    ys = tf.clip_by_value(ys, 0.0, tf.cast(h - 1, tf.float32))

    indices = tf.stack([tf.cast(tf.round(ys), tf.int32), tf.cast(tf.round(xs), tf.int32)], axis=-1)
    warped = tf.gather_nd(img, indices)
    return warped


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Index MURA (elbow/hand/shoulder) and print split summary.")
    ap.add_argument("--mura-root", required=True, help="Path to the MURA-v1.1 folder (contains train/, valid/)")
    ap.add_argument("--out-csv", default="mura_ehr_index.csv")
    args = ap.parse_args()

    df = index_mura(args.mura_root)
    df = make_patient_level_split(df)
    print_split_summary(df)
    df.to_csv(args.out_csv, index=False)
    print(f"\nFull index + split written to {args.out_csv}")
