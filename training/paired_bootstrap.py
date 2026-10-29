"""
paired_bootstrap.py -- is "GRAZ-pretrained beats ImageNet-only" statistically
supported, or within test-set noise? Study-level, paired bootstrap (10,000
resamples of test studies, same studies scored for both models) + exact
McNemar test, per region, then pooled across the three regions per backbone
(stratified resampling). Test-set sampling noise only -- each config was
trained once, so this does NOT capture seed-to-seed training variance.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from study_level_eval import aggregate, study_ids_for

REGIONS = ["XR_ELBOW", "XR_HAND", "XR_SHOULDER"]
PAIRS = {
    "DenseNet169": ("strong_DenseNet169", "graz_pretrained_densenet169"),
    "DenseNet121": ("strong_DenseNet121", "graz_pretrained_densenet121"),
    "EfficientNetV2B0": ("strong_EfficientNetV2B0", "graz_pretrained_efficientnetv2b0"),
    "ViT-B16": ("vit_b16", "graz_pretrained_vit_b16"),
}


def study_correct(model_dir, test_sid, test_y_img):
    tp = np.load(os.path.join(model_dir, "test_probs.npy"))[:, 1]
    tl = np.load(os.path.join(model_dir, "test_labels.npy"))
    assert len(tp) == len(test_sid) and np.array_equal(tl, test_y_img), (
        f"{model_dir}: saved rows/labels do not match the index row order")
    sp, sy = aggregate(tp, test_sid, test_y_img)
    return ((sp >= 0.5).astype(int) == sy).astype(int)


def mcnemar_exact(base, new):
    b = int(((base == 1) & (new == 0)).sum())
    c = int(((base == 0) & (new == 1)).sum())
    if b + c == 0:
        return b, c, 1.0
    try:
        from scipy.stats import binomtest
        p = float(binomtest(min(b, c), b + c, 0.5).pvalue)
    except Exception:
        from scipy.stats import binom_test
        p = float(binom_test(min(b, c), b + c, 0.5))
    return b, c, p


def two_sided_p(diffs):
    B = len(diffs)
    le = (np.sum(diffs <= 0) + 1) / (B + 1)
    ge = (np.sum(diffs >= 0) + 1) / (B + 1)
    return float(min(1.0, 2 * min(le, ge)))


def fmt(d):
    return f"{100 * d:+.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    df = pd.read_csv(args.index_csv)
    out = {"n_boot": args.n_boot, "per_region": [], "pooled": {}}
    pooled_inputs = {}

    print(f"{'Backbone':<17}{'Region':<13}{'N':>4} {'base':>7} {'new':>7} {'diff pts':>9}"
          f"  {'95% CI (pts)':<17}{'boot p':>8}{'McNemar p':>11}  b/c")
    for region in REGIONS:
        test_sid, test_y = study_ids_for(df, region, "test")
        for bb, (base_suffix, new_suffix) in PAIRS.items():
            bdir = os.path.join(args.results_dir, f"tier2_{region}_{base_suffix}")
            ndir = os.path.join(args.results_dir, f"tier2_{region}_{new_suffix}")
            if not (os.path.exists(os.path.join(bdir, "test_probs.npy"))
                    and os.path.exists(os.path.join(ndir, "test_probs.npy"))):
                continue
            base = study_correct(bdir, test_sid, test_y)
            new = study_correct(ndir, test_sid, test_y)
            n = len(base)
            idx = rng.randint(0, n, size=(args.n_boot, n))
            diffs = new[idx].mean(axis=1) - base[idx].mean(axis=1)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            p_boot = two_sided_p(diffs)
            b, c, p_mc = mcnemar_exact(base, new)
            d = float(new.mean() - base.mean())
            print(f"{bb:<17}{region:<13}{n:>4} {100*base.mean():>6.2f}% {100*new.mean():>6.2f}% "
                  f"{fmt(d):>9}  [{fmt(lo)}, {fmt(hi)}]".ljust(80) + f"{p_boot:>8.4f}{p_mc:>11.4f}  {b}/{c}")
            out["per_region"].append({"backbone": bb, "region": region, "n_studies": n,
                                      "baseline_acc": float(base.mean()), "new_acc": float(new.mean()),
                                      "diff": d, "ci95": [float(lo), float(hi)],
                                      "p_bootstrap": p_boot, "p_mcnemar": p_mc, "b": b, "c": c})
            pooled_inputs.setdefault(bb, []).append((base, new))

    print("\nPOOLED across regions (stratified resampling per region):")
    print(f"{'Backbone':<17}{'regions':>8}{'N':>6} {'diff pts':>9}  {'95% CI (pts)':<17}{'boot p':>8}")
    for bb, lst in pooled_inputs.items():
        N = sum(len(b) for b, _ in lst)
        obs = (sum(n.sum() for _, n in lst) - sum(b.sum() for b, _ in lst)) / N
        diffs = np.zeros(args.n_boot)
        for base, new in lst:
            n = len(base)
            idx = rng.randint(0, n, size=(args.n_boot, n))
            diffs += new[idx].sum(axis=1) - base[idx].sum(axis=1)
        diffs /= N
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p = two_sided_p(diffs)
        print(f"{bb:<17}{len(lst):>8}{N:>6} {fmt(obs):>9}  [{fmt(lo)}, {fmt(hi)}]".ljust(62) + f"{p:>8.4f}")
        out["pooled"][bb] = {"n_regions": len(lst), "n_studies": int(N), "diff": float(obs),
                             "ci95": [float(lo), float(hi)], "p_bootstrap": p}

    if not out["per_region"]:
        print("No baseline/GRAZ pairs with saved probabilities were found under --results-dir.")
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
