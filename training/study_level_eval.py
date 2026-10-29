"""
study_level_eval.py -- re-scores already-trained Tier 2 models at the STUDY
level instead of the image level, with no retraining.

MURA labels belong to a study (a folder like patient00123/study1_positive
holding 2-5 views). Published MURA work averages per-view predictions of a
study into one decision. Our training scripts report per-IMAGE accuracy.

Reads val_probs.npy/test_probs.npy/val_labels.npy/test_labels.npy from each
--model-dir (written by training scripts run with --save-probs). Nothing is
tuned on the test set.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def study_ids_for(df, region, split):
    sub = df[(df["region"] == region) & (df["split"] == split)].reset_index(drop=True)
    return sub["image_path"].apply(os.path.dirname).values, sub["fracture_label"].values


def aggregate(probs_pos, study_ids, labels):
    d = pd.DataFrame({"s": study_ids, "p": probs_pos, "y": labels})
    g = d.groupby("s").agg(p=("p", "mean"), y=("y", "first"), y_nuniq=("y", "nunique"))
    assert (g["y_nuniq"] == 1).all(), "A study has mixed labels -- grouping by folder is wrong."
    return g["p"].values, g["y"].values.astype(int)


def acc(p, y, thr=0.5):
    return float(((p >= thr).astype(int) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--model-dir", action="append", required=True)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.index_csv)
    val_sid, val_y_img = study_ids_for(df, args.region, "val")
    test_sid, test_y_img = study_ids_for(df, args.region, "test")
    print(f"{args.region}: val {len(val_sid)} images / {len(set(val_sid))} studies, "
          f"test {len(test_sid)} images / {len(set(test_sid))} studies")

    names, val_p, test_p = [], [], []
    for d in args.model_dir:
        vp = np.load(os.path.join(d, "val_probs.npy"))[:, 1]
        tp = np.load(os.path.join(d, "test_probs.npy"))[:, 1]
        vl = np.load(os.path.join(d, "val_labels.npy"))
        tl = np.load(os.path.join(d, "test_labels.npy"))
        assert len(vp) == len(val_sid) and len(tp) == len(test_sid), (
            f"{d}: prob rows ({len(vp)}/{len(tp)}) != index rows ({len(val_sid)}/{len(test_sid)}). "
            f"Row order cannot be trusted -- do not use this model here.")
        assert np.array_equal(vl, val_y_img) and np.array_equal(tl, test_y_img), (
            f"{d}: saved labels do not match index row order.")
        names.append(os.path.basename(d.rstrip("/")))
        val_p.append(vp)
        test_p.append(tp)

    result = {"region": args.region, "n_test_images": int(len(test_sid)),
              "n_test_studies": int(len(set(test_sid))), "models": {}}

    print("\nModel                                         image-acc  study-acc")
    for n, tp in zip(names, test_p):
        img_acc = acc(tp, test_y_img)
        sp, sy = aggregate(tp, test_sid, test_y_img)
        st_acc = acc(sp, sy)
        result["models"][n] = {"image_level": img_acc, "study_level": st_acc}
        print(f"{n:<45} {img_acc:9.4f} {st_acc:10.4f}")

    if len(test_p) > 1:
        avg = np.mean(np.stack(test_p), axis=0)
        sp, sy = aggregate(avg, test_sid, test_y_img)
        result["average_ensemble"] = {"image_level": acc(avg, test_y_img), "study_level": acc(sp, sy)}

        meta = LogisticRegression(max_iter=1000).fit(np.column_stack(val_p), val_y_img)
        stk = meta.predict_proba(np.column_stack(test_p))[:, 1]
        sp, sy = aggregate(stk, test_sid, test_y_img)
        result["stacked_ensemble"] = {"image_level": acc(stk, test_y_img), "study_level": acc(sp, sy)}

        print(f"{'AVERAGE ensemble':<45} {result['average_ensemble']['image_level']:9.4f} "
              f"{result['average_ensemble']['study_level']:10.4f}")
        print(f"{'STACKED ensemble':<45} {result['stacked_ensemble']['image_level']:9.4f} "
              f"{result['stacked_ensemble']['study_level']:10.4f}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
