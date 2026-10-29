"""
Gate test, step 2: train_tier2_strong.py exactly as before, but every image is replaced by its attention crop
(box from attention_boxes.py), cut from a higher-resolution load and resized to --img-size.
All other CLI arguments are forwarded unchanged to train_tier2_strong.main().
"""
import argparse, sys
import numpy as np
import tensorflow as tf


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--hi-res", type=int, default=448)
    known, rest = ap.parse_known_args()

    z = np.load(known.boxes, allow_pickle=True)
    box_by_path = {str(p): b for p, b in zip(z["paths"], z["boxes"])}
    print(f"Loaded {len(box_by_path)} crop boxes (frac={float(z['frac'])}) from {known.boxes}; hi-res load {known.hi_res}px")

    import preprocessing_v2 as P
    import train_tier2_strong as T

    def build_crop(df, label_col, batch_size=32, augment=False, use_autocrop=True, use_clahe=True,
                   shuffle=False, img_size=224):
        from data_utils import _augment_image, RNG_SEED   # read at call time so --seed takes effect
        paths = df["image_path"].values
        missing = [p for p in paths if str(p) not in box_by_path]
        assert not missing, f"{len(missing)} images have no crop box, e.g. {missing[0]}"
        boxes = np.stack([box_by_path[str(p)] for p in paths]).astype(np.float32)
        labels = df[label_col].values.astype(np.int64)

        def _load(path_t, label_t, box_t):
            path = path_t.numpy().decode("utf-8")
            y0, x0, y1, x1 = [float(v) for v in box_t.numpy()]
            img = P.load_and_preprocess_v2(path, known.hi_res, use_autocrop=use_autocrop, use_clahe=use_clahe)
            H, W = img.shape[:2]
            a, b = int(round(y0 * H)), max(int(round(y1 * H)), int(round(y0 * H)) + 8)
            c, d = int(round(x0 * W)), max(int(round(x1 * W)), int(round(x0 * W)) + 8)
            crop = tf.image.resize(img[a:b, c:d], (img_size, img_size)).numpy().astype(np.float32)
            return crop, label_t

        def _tf_load(path_t, label_t, box_t):
            im, lab = tf.py_function(_load, [path_t, label_t, box_t], [tf.float32, tf.int64])
            im.set_shape([img_size, img_size, 3])
            lab.set_shape([])
            return im, lab

        ds = tf.data.Dataset.from_tensor_slices((paths, labels, boxes))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df), seed=RNG_SEED, reshuffle_each_iteration=True)
        ds = ds.map(_tf_load, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(lambda x, y: (_augment_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    T.build_tf_dataset_v2 = build_crop     # train_one_config looks this name up in its own module globals
    sys.argv = [sys.argv[0]] + rest
    T.main()


if __name__ == "__main__":
    main()
