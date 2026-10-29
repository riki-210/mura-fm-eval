import argparse, glob, json, os
import numpy as np
import pandas as pd
from study_level_eval import aggregate, study_ids_for

REGIONS = ["XR_ELBOW", "XR_HAND", "XR_SHOULDER"]


def study_corr(p_img, sid, y_img):
    sp, sy = aggregate(p_img, sid, y_img)
    return ((sp >= 0.5).astype(float) == sy).astype(float)


def two_sided_p(d):
    B = len(d)
    return float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (B + 1), (np.sum(d >= 0) + 1) / (B + 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--ref-suffix", default="graz_pretrained_densenet169")
    ap.add_argument("--seed-glob", default="graz_d169_320_s*")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()
    rng = np.random.RandomState(0)
    df = pd.read_csv(args.index_csv)
    per = {}
    out = {"regions": {}}

    for R in REGIONS:
        sid, y = study_ids_for(df, R, "test")
        rdir = os.path.join(args.results_dir, f"tier2_{R}_{args.ref_suffix}")
        sdirs = sorted(glob.glob(os.path.join(args.results_dir, f"tier2_{R}_{args.seed_glob}")))
        sdirs = [d for d in sdirs if os.path.exists(os.path.join(d, "test_probs_noaug.npy"))]
        if not sdirs or not os.path.exists(os.path.join(rdir, "test_probs.npy")):
            print(f"{R}: reference or finished 320px seed runs missing, skipping"); continue
        ref = np.load(os.path.join(rdir, "test_probs.npy"))[:, 1]
        rows, plain_p, tta_p = [], [], []
        for d in sdirs:
            pp = np.load(os.path.join(d, "test_probs_noaug.npy"))[:, 1]
            pt = np.load(os.path.join(d, "test_probs.npy"))[:, 1]
            assert len(pp) == len(sid), f"{d}: rows do not match index"
            plain_p.append(pp); tta_p.append(pt)
            rows.append({"run": os.path.basename(d),
                         "img_plain": float(((pp >= .5) == y).mean()), "img_tta": float(((pt >= .5) == y).mean()),
                         "study_plain": float(study_corr(pp, sid, y).mean()), "study_tta": float(study_corr(pt, sid, y).mean())})
        ens_plain, ens_tta = np.mean(plain_p, 0), np.mean(tta_p, 0)
        ref_row = {"img": float(((ref >= .5) == y).mean()), "study": float(study_corr(ref, sid, y).mean())}
        print(f"\n=== {R}  ({len(sdirs)} seeds; {len(set(sid))} test studies) ===")
        print(f"{'run':<34}{'img no-TTA':>11}{'img TTA':>9}{'study no-TTA':>14}{'study TTA':>11}")
        print(f"{'224px reference (1 run)':<34}{100*ref_row['img']:>10.2f}%{'':>9}{100*ref_row['study']:>13.2f}%")
        for r in rows:
            print(f"{r['run']:<34}{100*r['img_plain']:>10.2f}%{100*r['img_tta']:>8.2f}%{100*r['study_plain']:>13.2f}%{100*r['study_tta']:>10.2f}%")
        m = {k: np.array([r[k] for r in rows]) for k in ("img_plain", "img_tta", "study_plain", "study_tta")}
        print(f"{'mean +- sd over seeds':<34}" + f"{100*m['img_plain'].mean():>6.2f}+-{100*m['img_plain'].std(ddof=1) if len(rows)>1 else 0:.2f}"
              f"{100*m['img_tta'].mean():>7.2f}%" + f"{100*m['study_plain'].mean():>10.2f}+-{100*m['study_plain'].std(ddof=1) if len(rows)>1 else 0:.2f}"
              f"{100*m['study_tta'].mean():>8.2f}%")
        e_img_p, e_img_t = float(((ens_plain >= .5) == y).mean()), float(((ens_tta >= .5) == y).mean())
        e_st_p, e_st_t = study_corr(ens_plain, sid, y), study_corr(ens_tta, sid, y)
        print(f"{'seed-ensemble':<34}{100*e_img_p:>10.2f}%{100*e_img_t:>8.2f}%{100*e_st_p.mean():>13.2f}%{100*e_st_t.mean():>10.2f}%")
        per[R] = {"ref": study_corr(ref, sid, y),
                  "plain_mean": np.mean([study_corr(p, sid, y) for p in plain_p], 0),
                  "tta_mean": np.mean([study_corr(p, sid, y) for p in tta_p], 0),
                  "ens_tta": e_st_t}
        out["regions"][R] = {"ref224": ref_row, "seeds": rows,
                             "seed_ensemble": {"img_plain": e_img_p, "img_tta": e_img_t,
                                               "study_plain": float(e_st_p.mean()), "study_tta": float(e_st_t.mean())}}

    if per:
        print("\nPOOLED over regions (paired bootstrap, stratified; study level):")
        print(f"{'comparison':<62}{'diff pts':>9}  {'95% CI':<17}{'p':>8}")
        comps = [("(a) resolution: 320 mean-of-seeds no-TTA  vs  224 single", "plain_mean", "ref"),
                 ("(b) TTA: 320 mean-of-seeds TTA  vs  320 no-TTA", "tta_mean", "plain_mean"),
                 ("(c) total: 3-seed ensemble + TTA  vs  224 single", "ens_tta", "ref")]
        out["pooled"] = {}
        for label, kn, kb in comps:
            N = sum(len(v[kn]) for v in per.values())
            obs = sum(v[kn].sum() - v[kb].sum() for v in per.values()) / N
            diffs = np.zeros(args.n_boot)
            for v in per.values():
                n = len(v[kn]); idx = rng.randint(0, n, size=(args.n_boot, n))
                diffs += v[kn][idx].sum(1) - v[kb][idx].sum(1)
            diffs /= N
            lo, hi = np.percentile(diffs, [2.5, 97.5]); p = two_sided_p(diffs)
            print(f"{label:<62}{100*obs:>+8.2f}  [{100*lo:+.2f}, {100*hi:+.2f}]".ljust(98) + f"{p:>8.4f}")
            out["pooled"][label] = {"diff": float(obs), "ci95": [float(lo), float(hi)], "p": p}
    if args.output_json:
        json.dump(out, open(args.output_json, "w"), indent=2); print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
