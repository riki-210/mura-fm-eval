"""
Paired study-level comparison of two groups of runs (glob per region; per-study correctness averaged over runs).
Example: compare_runs.py --index-csv ... --results-dir ... --a "graz_d169_crop60_s1" --b "graz_d169_224_s*"
"""
import argparse, glob, os
import numpy as np
import pandas as pd
from study_level_eval import aggregate, study_ids_for

REGIONS = ["XR_ELBOW", "XR_HAND", "XR_SHOULDER"]


def study_corr(p, sid, y):
    sp, sy = aggregate(p, sid, y)
    return ((sp >= 0.5).astype(float) == sy).astype(float)


def group(results, R, pattern, sid, y, noaug):
    dirs = []
    for d in sorted(glob.glob(os.path.join(results, f"tier2_{R}_{pattern}"))):
        f = "test_probs_noaug.npy" if noaug and os.path.exists(os.path.join(d, "test_probs_noaug.npy")) else "test_probs.npy"
        if os.path.exists(os.path.join(d, f)):
            dirs.append(os.path.join(d, f))
    if not dirs:
        return None, 0, None
    P = [np.load(f)[:, 1] for f in dirs]
    for p in P:
        assert len(p) == len(sid), "rows do not match index"
    return np.mean([study_corr(p, sid, y) for p in P], 0), len(dirs), float(np.mean([((p >= .5) == y).mean() for p in P]))


def pval(d):
    B = len(d)
    return float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (B + 1), (np.sum(d >= 0) + 1) / (B + 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--a", required=True, help="glob suffix of group A, e.g. graz_d169_crop60_s1")
    ap.add_argument("--b", required=True, help="glob suffix of group B (reference)")
    ap.add_argument("--noaug", action="store_true", help="use test_probs_noaug.npy (no TTA) where present")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    rng = np.random.RandomState(0)
    df = pd.read_csv(args.index_csv)
    per = {}
    print(f"A = {args.a}   B = {args.b}   (study-level accuracy; diff = A - B)")
    print(f"{'region':<13}{'runs A/B':>9}{'img A':>8}{'img B':>8}{'study A':>9}{'study B':>9}{'diff':>8}  {'95% CI':<16}{'p':>7}")
    for R in REGIONS:
        sid, y = study_ids_for(df, R, "test")
        a, na, ia = group(args.results_dir, R, args.a, sid, y, args.noaug)
        b, nb, ib = group(args.results_dir, R, args.b, sid, y, args.noaug)
        if a is None or b is None:
            print(f"{R:<13}  missing runs (A:{na} B:{nb}), skipped"); continue
        n = len(a)
        idx = rng.randint(0, n, size=(args.n_boot, n))
        diffs = a[idx].mean(1) - b[idx].mean(1)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"{R:<13}{f'{na}/{nb}':>9}{100*ia:>7.2f}%{100*ib:>7.2f}%{100*a.mean():>8.2f}%{100*b.mean():>8.2f}%"
              f"{100*(a.mean()-b.mean()):>+8.2f}  [{100*lo:+.2f},{100*hi:+.2f}]{pval(diffs):>9.4f}")
        per[R] = (a, b)
    if per:
        N = sum(len(a) for a, _ in per.values())
        obs = sum(a.sum() - b.sum() for a, b in per.values()) / N
        diffs = np.zeros(args.n_boot)
        for a, b in per.values():
            n = len(a); idx = rng.randint(0, n, size=(args.n_boot, n))
            diffs += a[idx].sum(1) - b[idx].sum(1)
        diffs /= N
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"{'POOLED':<13}{'':>9}{'':>8}{'':>8}{'':>9}{'':>9}{100*obs:>+8.2f}  [{100*lo:+.2f},{100*hi:+.2f}]{pval(diffs):>9.4f}")


if __name__ == "__main__":
    main()
