"""Writes randexcl_<REGION>.csv: random studies (same count as the confident-learning flags) from the region's train+val data.
Control for the cleaned-label retraining: removes as much data as the flagged set, but not the flagged studies. TEST IS NEVER TOUCHED."""
import os, numpy as np, pandas as pd
H = os.path.expanduser("~/mediinsight_project")
idx = pd.read_csv(f"{H}/data/mura_ehr_index.csv")
idx["study"] = idx.image_path.map(lambda p: os.path.dirname(str(p)))
out = f"{H}/results/label_cleaning"
for R in ("XR_ELBOW", "XR_HAND", "XR_SHOULDER"):
    flagged = pd.read_csv(f"{out}/exclude_{R}.csv").image_path.astype(str)
    n = flagged.map(os.path.dirname).nunique()
    pool = idx[(idx.region == R) & (idx.split.isin(["train", "val"]))]
    st = np.sort(pool.study.unique())
    pick = set(np.random.default_rng(0).choice(st, size=n, replace=False))
    rows = pool[pool.study.isin(pick)][["image_path"]]
    rows.to_csv(f"{out}/randexcl_{R}.csv", index=False)
    print(R, "flagged studies", n, "random studies", len(pick), "images", len(rows))
