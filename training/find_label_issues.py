"""
find_label_issues.py -- confident-learning style detection of mislabelled MURA STUDIES from out-of-fold probabilities
(train_fm_cv.py). MURA labels belong to a study (folder), so view probabilities are averaged per study first.
A study labelled j is flagged when the OOF model is confident it belongs to the other class:
    p(other) >= t_other, where t_c = mean OOF p(c) over studies labelled c   (Northcutt et al., confident learning)
At most --max-frac of the region's studies are flagged (lowest self-confidence first). TEST DATA IS NEVER TOUCHED.
Writes label_issues_<REGION>.csv (ranked, human-reviewable) and exclude_<REGION>.csv (image_path of flagged studies).
No torch needed.
"""
import argparse, os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-dir", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-frac", type=float, default=0.10)
    args = ap.parse_args()

    pool = pd.read_csv(os.path.join(args.cv_dir, "pooled_index.csv"))
    oof = np.load(os.path.join(args.cv_dir, "oof_probs.npy"))
    assert len(pool) == len(oof)
    pool["p1"] = oof[:, 1]
    pool["study"] = pool["image_path"].map(lambda p: os.path.dirname(str(p)))
    g = pool.groupby("study")
    st = pd.DataFrame({"label": g["fracture_label"].agg(lambda s: int(round(s.mean()))),
                       "label_mixed": g["fracture_label"].agg(lambda s: s.nunique() > 1),
                       "n_views": g.size(), "p_fracture": g["p1"].mean(), "fold": g["fold"].first()}).reset_index()
    print(f"{args.region}: {len(st)} studies, {int(st['label_mixed'].sum())} with mixed view labels (should be 0)")
    y, p1 = st["label"].values, st["p_fracture"].values
    st["self_conf"] = np.where(y == 1, p1, 1 - p1)
    t1 = p1[y == 1].mean()            # mean p(fracture) among studies labelled fracture
    t0 = (1 - p1[y == 0]).mean()      # mean p(normal) among studies labelled normal
    cand = ((y == 0) & (p1 >= t1)) | ((y == 1) & ((1 - p1) >= t0))
    st["candidate"] = cand
    st = st.sort_values("self_conf").reset_index(drop=True)
    st["rank"] = np.arange(1, len(st) + 1)
    cap = int(args.max_frac * len(st))
    # cap = max number of flagged studies; most suspicious (lowest self-confidence) candidates first
    st["flagged"] = False
    st.loc[st.index[st["candidate"]][:cap], "flagged"] = True

    acc = float(((p1 >= 0.5).astype(int) == y).mean())
    print(f"OOF study-level accuracy {acc:.4f}  thresholds t_fracture={t1:.3f} t_normal={t0:.3f}")
    print(f"confident-learning candidates {int(cand.sum())} ({cand.mean():.1%}); flagged after cap {int(st['flagged'].sum())} "
          f"({st['flagged'].mean():.1%})")
    for lab, name in ((0, "labelled normal, model says fracture"), (1, "labelled fracture, model says normal")):
        print(f"   {name}: {int((st['flagged'] & (st['label'] == lab)).sum())}")
    os.makedirs(args.out_dir, exist_ok=True)
    st.to_csv(os.path.join(args.out_dir, f"label_issues_{args.region}.csv"), index=False)
    bad = set(st.loc[st["flagged"], "study"])
    pool.loc[pool["study"].isin(bad), ["image_path"]].to_csv(os.path.join(args.out_dir, f"exclude_{args.region}.csv"), index=False)
    print(f"wrote label_issues_{args.region}.csv and exclude_{args.region}.csv "
          f"({int(pool['study'].isin(bad).sum())} images) in {args.out_dir}")


if __name__ == "__main__":
    main()
