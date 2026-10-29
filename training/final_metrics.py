"""
final_metrics.py -- paper/thesis result tables from the saved predictions (no GPU, no retraining).
Study level (MURA labels are per study): view probabilities are averaged per study folder, then models are averaged.
For each region (elbow/hand/shoulder) and POOLED over the three regions, for each model family and two ensembles:
  accuracy, sensitivity, specificity, PPV, NPV, F1, balanced accuracy, Cohen's kappa, AUC, each with a 95% percentile
  bootstrap CI (test STUDIES resampled; pooled = resampled within each region). Threshold 0.5, and -- for every group --
  the Youden-J threshold chosen on VALIDATION studies only and applied to test. Nothing is tuned on test.
Writes final_metrics.md, final_metrics.csv, roc_curves.png (if matplotlib is available) into --out-dir.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

REGIONS = ["XR_ELBOW", "XR_HAND", "XR_SHOULDER"]
S3 = (1, 2, 3)
FAM = {
    "DINOv2-base (image-level training)": [f"dinov2_base_s{s}" for s in S3],
    "DINOv2-base (study-level MIL)": [f"dinov2_mil2_s{s}" for s in S3],
    "DINOv2-base (all-body-parts training)": [f"dinov2_all7_s{s}" for s in S3],
    "GRAZ-pretrained DenseNet169": [f"graz_d169_224_s{s}" for s in S3],
}
DINO_ALL = ([f"dinov2_base_s{s}" for s in S3] + [f"dinov2_base_lr1e5_s{s}" for s in S3] + [f"dinov2_large_s{s}" for s in S3]
            + [f"rad_dino_s{s}" for s in S3] + [f"dinov2_mil2_s{s}" for s in S3])
ENS = {
    "Ensemble A: DINOv2 family (15 runs, average)": DINO_ALL,
    "Ensemble C: A plus all-body-parts (18 runs, average)": DINO_ALL + [f"dinov2_all7_s{s}" for s in S3],
    "Ensemble B: all 18 runs (adds GRAZ-DenseNet169), average": DINO_ALL + [f"graz_d169_224_s{s}" for s in S3],
}
GROUPS = {**FAM, **ENS}


def study_table(dsplit):
    st = dsplit["image_path"].map(lambda p: os.path.dirname(str(p))).values
    return st


def load_probs(results, region, run, split, n_rows):
    f = os.path.join(results, f"tier2_{region}_{run}", f"{split}_probs.npy")
    if not os.path.exists(f):
        return None
    a = np.load(f)
    assert len(a) == n_rows, f"{f}: {len(a)} rows, index has {n_rows}"
    return a[:, 1]


def to_study(dsplit, p_rows):
    s = pd.DataFrame({"study": study_table(dsplit), "p": p_rows, "y": dsplit["fracture_label"].values})
    g = s.groupby("study", sort=False)
    return g["p"].mean().values, g["y"].first().values


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    n = len(y)
    acc = (tp + tn) / n
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    return {"accuracy": acc, "sensitivity": sens, "specificity": spec, "ppv": tp / max(tp + fp, 1),
            "npv": tn / max(tn + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1), "balanced_acc": (sens + spec) / 2,
            "kappa": (acc - pe) / (1 - pe) if pe < 1 else float("nan"),
            "auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")}


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    k = int(np.argmax(tpr - fpr))
    t = thr[k]
    return float(np.clip(t, 0.0, 1.0)) if np.isfinite(t) else 0.5


def boot_ci(parts, n_boot, rng):
    """parts = list of (y, p, thr) per region; resample studies within each region; returns {metric: (lo, hi)}."""
    vals = {k: [] for k in metrics(parts[0][0], parts[0][1], parts[0][2])}
    for _ in range(n_boot):
        ys, ps, ts = [], [], []
        for y, p, t in parts:
            i = rng.integers(0, len(y), len(y)); ys.append(y[i]); ps.append(p[i]); ts.append(np.full(len(i), t))
        m = metrics(np.concatenate(ys), np.concatenate(ps), np.concatenate(ts))
        for k, v in m.items():
            vals[k].append(v)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.index_csv)

    data = {}        # (group, region) -> dict(test=(y,p), val=(y,p), n_runs, run_acc)
    for R in REGIONS:
        reg = df[df["region"] == R]
        parts = {sp: reg[reg["split"] == sp] for sp in ("val", "test")}
        cache = {}
        for gname, runs in GROUPS.items():
            got = {"val": [], "test": []}; run_acc = []
            for run in runs:
                ok = True
                for sp in ("val", "test"):
                    pr = load_probs(args.results_dir, R, run, sp, len(parts[sp]))
                    if pr is None:
                        ok = False; break
                    cache[(run, sp)] = to_study(parts[sp], pr)
                if not ok:
                    print(f"  [warn] missing {R} {run}", file=sys.stderr); continue
                for sp in ("val", "test"):
                    got[sp].append(cache[(run, sp)])
                p, y = cache[(run, "test")]          # to_study returns (probabilities, labels)
                run_acc.append(float(((p >= 0.5).astype(int) == y).mean()))
            if not got["test"]:
                continue
            ent = {"n_runs": len(got["test"]), "run_acc": run_acc}
            for sp in ("val", "test"):
                y = got[sp][0][1]
                for _, yy in got[sp]:
                    assert (yy == y).all()
                ent[sp] = (y, np.mean([p for p, _ in got[sp]], axis=0))
            data[(gname, R)] = ent
    rows = []
    for gname in GROUPS:
        if not all((gname, R) in data for R in REGIONS):
            print(f"[skip] {gname}: incomplete", file=sys.stderr); continue
        for mode in ("thr0.5", "val-Youden"):
            thr_by = {R: (0.5 if mode == "thr0.5" else youden_threshold(*data[(gname, R)]["val"])) for R in REGIONS}
            for R in REGIONS + ["POOLED"]:
                if R == "POOLED":
                    parts = [(*data[(gname, r)]["test"], thr_by[r]) for r in REGIONS]   # each region keeps its own threshold
                    point = metrics(np.concatenate([y for y, _, _ in parts]), np.concatenate([p for _, p, _ in parts]),
                                    np.concatenate([np.full(len(y), t) for y, _, t in parts]))
                    ci = boot_ci(parts, args.n_boot, rng)
                    n = sum(len(y) for y, _, _ in parts); thr_txt = "/".join(f"{thr_by[r]:.2f}" for r in REGIONS)
                else:
                    y, p = data[(gname, R)]["test"]
                    point = metrics(y, p, thr_by[R]); ci = boot_ci([(y, p, thr_by[R])], args.n_boot, rng)
                    n = len(y); thr_txt = f"{thr_by[R]:.3f}"
                ra = data[(gname, R)]["run_acc"] if R != "POOLED" else []
                rows.append({"group": gname, "mode": mode, "region": R, "n_studies": n, "threshold": thr_txt,
                             "n_runs": data[(gname, REGIONS[0])]["n_runs"],
                             "run_acc_mean": float(np.mean(ra)) if len(ra) else np.nan,
                             "run_acc_sd": float(np.std(ra, ddof=1)) if len(ra) > 1 else np.nan,
                             **{k: point[k] for k in point}, **{k + "_lo": ci[k][0] for k in ci}, **{k + "_hi": ci[k][1] for k in ci}})
            print(f"done {gname} [{mode}]", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out_dir, "final_metrics.csv"), index=False)

    cell = lambda r, k: f"{100 * r[k]:.1f} [{100 * r[k + '_lo']:.1f}-{100 * r[k + '_hi']:.1f}]"
    cellf = lambda r, k: f"{r[k]:.3f} [{r[k + '_lo']:.3f}-{r[k + '_hi']:.3f}]"
    md = ["# Final study-level results (test set, 95% bootstrap CIs over test studies)", "",
          "Study-level: view probabilities averaged per study; models averaged. `thr0.5` = fixed 0.5; `val-Youden` = threshold chosen on validation studies only.", ""]
    for mode in ("thr0.5", "val-Youden"):
        md += [f"## Threshold: {mode}", ""]
        for R in REGIONS + ["POOLED"]:
            sub = out[(out["mode"] == mode) & (out["region"] == R)]
            md += [f"### {R}  (n = {int(sub['n_studies'].iloc[0])} test studies)", "",
                   "| Model | Acc % | Sens % | Spec % | PPV % | NPV % | F1 % | AUC | Kappa | Thr | Run acc mean+-SD |",
                   "|---|---|---|---|---|---|---|---|---|---|---|"]
            for _, r in sub.iterrows():
                sd = f"{100 * r['run_acc_mean']:.1f}+-{100 * r['run_acc_sd']:.1f}" if r["run_acc_sd"] == r["run_acc_sd"] else "-"
                md.append(f"| {r['group']} | {cell(r, 'accuracy')} | {cell(r, 'sensitivity')} | {cell(r, 'specificity')} | {cell(r, 'ppv')} | "
                          f"{cell(r, 'npv')} | {cell(r, 'f1')} | {cellf(r, 'auc')} | {r['kappa']:.3f} | {r['threshold']} | {sd} |")
            md.append("")
    open(os.path.join(args.out_dir, "final_metrics.md"), "w").write("\n".join(md))
    print("wrote final_metrics.md / final_metrics.csv in", args.out_dir)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
        for ax, R in zip(axs, REGIONS):
            for gname in GROUPS:
                if (gname, R) not in data:
                    continue
                y, p = data[(gname, R)]["test"]
                fpr, tpr, _ = roc_curve(y, p)
                ax.plot(fpr, tpr, lw=2 if gname.startswith("Ensemble") else 1.2,
                        label=f"{gname.split(':')[0] if gname.startswith('Ensemble') else gname} (AUC {roc_auc_score(y, p):.3f})")
            ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_title(R.replace("XR_", "").title() + " (study level, test)")
            ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity"); ax.legend(fontsize=7, loc="lower right")
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "roc_curves.png"), dpi=150)
        print("wrote roc_curves.png")
    except Exception as e:
        print("[warn] ROC figure skipped:", e, file=sys.stderr)


if __name__ == "__main__":
    main()
