"""
preprocessing_v2.py -- stronger preprocessing for the Tier 2 accuracy-push
round: auto-crop (removes the black borders/markers common in raw MURA
films, so the network sees bone rather than padding) and CLAHE contrast
enhancement (standard, well-evidenced for radiograph models, distinct from
the paper's own 4-stage enhancement.py pipeline -- this is a separate,
additional preprocessing option, not a replacement for that pipeline; both
can be compared).
"""

import cv2
import numpy as np


def autocrop_border(gray_u8: np.ndarray, thresh: int = 10, margin_frac: float = 0.02) -> np.ndarray:
    """
    Crops away near-black border regions (common in raw MURA films: black
    padding, corner markers/labels) by finding the bounding box of pixels
    above `thresh` intensity and cropping to it (plus a small margin so the
    crop doesn't clip the bone itself at the boundary). Falls back to the
    original image untouched if the detected box would be degenerate
    (e.g. a near-uniform image) -- never raises, always returns a valid image.
    """
    h, w = gray_u8.shape[:2]
    mask = gray_u8 > thresh
    if not mask.any():
        return gray_u8

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    margin_r = int((rmax - rmin) * margin_frac)
    margin_c = int((cmax - cmin) * margin_frac)
    rmin = max(0, rmin - margin_r)
    rmax = min(h - 1, rmax + margin_r)
    cmin = max(0, cmin - margin_c)
    cmax = min(w - 1, cmax + margin_c)

    # Guard against a degenerate (too-small) crop -- keep original if so.
    if (rmax - rmin) < 0.3 * h or (cmax - cmin) < 0.3 * w:
        return gray_u8

    return gray_u8[rmin:rmax + 1, cmin:cmax + 1]


def apply_clahe(gray_u8: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization -- standard radiograph contrast booster."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_u8)


def load_and_preprocess_v2(image_path: str, img_size: int, use_autocrop: bool = True,
                            use_clahe: bool = True) -> np.ndarray:
    """
    Loads one MURA PNG and applies (optionally) auto-cropping and CLAHE,
    then resizes to img_size x img_size and returns a (img_size, img_size, 3)
    float32 array in [0, 1], 3-channel-replicated -- same convention as
    data_utils.load_and_enhance, so it's a drop-in input for the same models.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if use_autocrop:
        img = autocrop_border(img)
    if use_clahe:
        img = apply_clahe(img)

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_f = img.astype(np.float32) / 255.0
    rgb = np.stack([img_f, img_f, img_f], axis=-1)
    return rgb.astype(np.float32)


def build_tf_dataset_v2(df, label_col: str, batch_size: int = 32, augment: bool = False,
                         use_autocrop: bool = True, use_clahe: bool = True,
                         shuffle: bool = False, img_size: int = 224):
    """
    Same shape/contract as data_utils.build_tf_dataset, but uses
    load_and_preprocess_v2 (autocrop + CLAHE) instead of the paper's 4-stage
    enhancement pipeline. Reuses data_utils._augment_image for augmentation
    so train-time augmentation stays identical/comparable across experiments.
    """
    import tensorflow as tf
    from data_utils import _augment_image, RNG_SEED

    paths = df["image_path"].values
    labels = df[label_col].values.astype(np.int64)

    def _load(path_tensor, label_tensor):
        path = path_tensor.numpy().decode("utf-8")
        img = load_and_preprocess_v2(path, img_size, use_autocrop=use_autocrop, use_clahe=use_clahe)
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
