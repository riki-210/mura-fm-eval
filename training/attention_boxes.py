"""
Gate test, step 1: label-free attention boxes from an already-trained fracture model.
"""
import argparse, os
import numpy as np
import pandas as pd
import tensorflow as tf
from preprocessing_v2 import build_tf_dataset_v2


def load_model(path, img_size):
    try:
        m = tf.keras.models.load_model(path, compile=False)
        print(f"Loaded full model from {path}")
        return m
    except Exception as e:  # noqa
        print(f"load_model failed ({type(e).__name__}: {str(e)[:120]}); rebuilding architecture + load_weights")
        from models_tier2_strong import build_strong_fracture_classifier
        m = build_strong_fracture_classifier("DenseNet169", img_size=img_size, num_classes=2, dropout=0.5,
                                             l2_reg=1e-4, freeze_backbone=False, domain_weights_path=None)
        m.load_weights(path)
        return m


def saliency_maps(model, ds, n_samples=4, noise=0.1, blur=21):
    maps = []
    for x, _ in ds:
        x = tf.cast(x, tf.float32)
        rng = tf.reduce_max(x, axis=[1, 2, 3], keepdims=True) - tf.reduce_min(x, axis=[1, 2, 3], keepdims=True)
        acc = tf.zeros(tf.shape(x)[:3])
        for _ in range(n_samples):
            xn = x + tf.random.normal(tf.shape(x), stddev=1.0) * noise * rng
            with tf.GradientTape() as tape:
                tape.watch(xn)
                p = model(xn, training=False)[:, 1]
            g = tape.gradient(p, xn)
            acc += tf.reduce_max(tf.abs(g), axis=-1)
        acc = acc / n_samples
        acc = tf.nn.avg_pool2d(acc[..., None], blur, 1, "SAME")[..., 0]
        maps.append(acc.numpy())
    return np.concatenate(maps, 0)


def boxes_from_maps(maps, frac):
    n, H, W = maps.shape
    flat = maps.reshape(n, -1).argmax(1)
    cy, cx = (flat // W) / H, (flat % W) / W
    y0 = np.clip(cy - frac / 2, 0, 1 - frac)
    x0 = np.clip(cx - frac / 2, 0, 1 - frac)
    return np.stack([y0, x0, y0 + frac, x0 + frac], 1).astype(np.float32)


def montage(df, boxes_by_path, img_size, out_png, n=12, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = df.sample(min(n, len(df)), random_state=seed)
    ds = build_tf_dataset_v2(sub, "fracture_label", batch_size=len(sub), augment=False, shuffle=False, img_size=img_size)
    x, _ = next(iter(ds))
    x = x.numpy()
    cols = 4
    rows = int(np.ceil(len(sub) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for ax in np.array(axs).reshape(-1):
        ax.axis("off")
    for i, (p, lab) in enumerate(zip(sub["image_path"], sub["fracture_label"])):
        ax = np.array(axs).reshape(-1)[i]
        im = x[i]
        ax.imshow((im - im.min()) / (im.max() - im.min() + 1e-8))
        y0, x0, y1, x1 = boxes_by_path[p]
        ax.add_patch(plt.Rectangle((x0 * img_size, y0 * img_size), (x1 - x0) * img_size, (y1 - y0) * img_size,
                                   fill=False, edgecolor="red", linewidth=2))
        ax.set_title(f"label={int(lab)}", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=90)
    print(f"Wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--frac", type=float, default=0.6, help="crop side as a fraction of the image side")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--montage-png", default=None)
    args = ap.parse_args()
    if os.path.exists(args.out_npz):
        print(f"{args.out_npz}: boxes exist, skipping box computation"); return
    tf.random.set_seed(0)

    df = pd.read_csv(args.index_csv)
    reg = df[df["region"] == args.region]
    assert len(reg), f"no rows for region {args.region}; region values: {sorted(df['region'].unique())}"
    model = load_model(args.model_path, args.img_size)

    all_paths, all_boxes = [], []
    for split in ("train", "val", "test"):
        d = reg[reg["split"] == split]
        assert len(d), f"no rows for split {split}; split values: {sorted(reg['split'].unique())}"
        ds = build_tf_dataset_v2(d, "fracture_label", args.batch_size, augment=False, shuffle=False, img_size=args.img_size)
        maps = saliency_maps(model, ds)
        assert len(maps) == len(d)
        all_paths += list(d["image_path"].values)
        all_boxes.append(boxes_from_maps(maps, args.frac))
        print(f"{args.region} {split}: {len(d)} images, mean box centre "
              f"({np.mean(all_boxes[-1][:, 0] + args.frac / 2):.2f}, {np.mean(all_boxes[-1][:, 1] + args.frac / 2):.2f})")
    boxes = np.concatenate(all_boxes, 0)
    np.savez(args.out_npz, paths=np.array(all_paths), boxes=boxes, frac=args.frac)
    print(f"Wrote {args.out_npz}  ({len(all_paths)} boxes)")
    if args.montage_png:
        bp = dict(zip(all_paths, boxes))
        try:
            montage(reg[reg["split"] == "test"], bp, args.img_size, args.montage_png)
        except Exception as e:  # montage is optional; never let it kill the job
            print(f"montage skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
