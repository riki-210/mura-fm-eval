"""
run_ablation_5config.py -- The 5-config hyperparameter ablation on the
SHOULDER subset only, per BoneFracture_Tier2_FINAL.pdf: dropout in
{0.5, 0.65, 0.8}, L2 in {0, 1e-4, 1e-3}, PCA on/off -- 5 of the 3x3x2=18
possible combinations were reported in the paper (not the full grid), with
the published best config being dropout=0.8, L2=1e-4, PCA=on -> 92.71% val
accuracy on shoulder.

Since the paper reports only 5 specific configurations rather than the full
3x3x2 grid, and the original code (and therefore the exact 5 chosen
combinations) is lost, this script runs a defensible default set of 5 that
brackets the reported best config and its immediate neighbours, so you can
confirm the best-config number (92.71%) and see the same directional trend
(PCA + higher dropout + moderate L2 helping) even if the exact secondary
configs don't match the original paper's table row-for-row:

    1. dropout=0.5, L2=0,     PCA=off   (baseline, no regularization added)
    2. dropout=0.5, L2=0,     PCA=on    (isolates PCA's contribution alone)
    3. dropout=0.8, L2=0,     PCA=on    (isolates dropout's added contribution)
    4. dropout=0.8, L2=1e-4,  PCA=on    (the paper's reported best config)
    5. dropout=0.8, L2=1e-3,  PCA=on    (checks if a stronger L2 helps or hurts)

If you locate the actual 5 configurations from any surviving note, figure
caption, or supplementary file, replace CONFIGS below with the exact list --
everything downstream (train_one_config) is agnostic to which 5 are chosen.

Usage:
    python run_ablation_5config.py --index-csv mura_ehr_index.csv \
        --output-dir results/ablation_shoulder
"""

import argparse
import os
import json

import pandas as pd

from train_tier2_alexnet import train_one_config

CONFIGS = [
    {"dropout": 0.5, "l2": 0.0,   "use_pca": False},
    {"dropout": 0.5, "l2": 0.0,   "use_pca": True},
    {"dropout": 0.8, "l2": 0.0,   "use_pca": True},
    {"dropout": 0.8, "l2": 1e-4,  "use_pca": True},   # paper's reported best config
    {"dropout": 0.8, "l2": 1e-3,  "use_pca": True},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--output-dir", default="results/ablation_shoulder")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-enhancement", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.index_csv)
    region_df = df[df["region"] == "XR_SHOULDER"]
    train_df = region_df[region_df["split"] == "train"].reset_index(drop=True)
    val_df = region_df[region_df["split"] == "val"].reset_index(drop=True)
    test_df = region_df[region_df["split"] == "test"].reset_index(drop=True)

    results = []
    for i, cfg in enumerate(CONFIGS, start=1):
        cfg_dir = os.path.join(
            args.output_dir,
            f"cfg{i}_dropout{cfg['dropout']}_l2{cfg['l2']}_pca{cfg['use_pca']}",
        )
        print(f"\n{'=' * 70}\nConfig {i}/{len(CONFIGS)}: {cfg}\n{'=' * 70}")

        # Reuse one PCA fit across the PCA-on configs for this region, rather
        # than refitting PCA from scratch for every PCA-on config -- the PCA
        # front-end doesn't change with dropout/L2, only the classifier head
        # does, so this saves a meaningful amount of redundant compute.
        pca_cache = os.path.join(args.output_dir, "shoulder_pca.joblib") if cfg["use_pca"] else None

        result = train_one_config(
            train_df, val_df, test_df, region="XR_SHOULDER",
            dropout=cfg["dropout"], l2_reg=cfg["l2"], use_pca=cfg["use_pca"],
            output_dir=cfg_dir, batch_size=args.batch_size, epochs=args.epochs,
            apply_enhancement=not args.no_enhancement, pca_cache_path=pca_cache,
        )
        result["config_id"] = i
        results.append(result)

        # Write incrementally so a partial run's results aren't lost.
        pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "ablation_5config_results.csv"), index=False)

    print("\n===== 5-CONFIG ABLATION SUMMARY (shoulder) =====")
    summary_df = pd.DataFrame(results)
    print(summary_df[["config_id", "dropout", "l2_reg", "use_pca", "val_accuracy", "test_accuracy"]].to_string(index=False))

    best = summary_df.loc[summary_df["val_accuracy"].idxmax()]
    print(f"\nBest config: #{int(best['config_id'])} "
          f"(dropout={best['dropout']}, L2={best['l2_reg']}, PCA={best['use_pca']}) "
          f"-> val_accuracy={best['val_accuracy']:.4f}")
    print("Compare this against the paper's reported 92.71% for "
          "dropout=0.8, L2=1e-4, PCA=on.")


if __name__ == "__main__":
    main()
