"""
collect_results_table.py -- scans a results directory for every Tier 2
summary.json this suite produces (original AlexNet, pretrained-AlexNet,
PCA-reconstruct, strong CNN backbones, ViT, hybrid, GRAZPEDWRI-DX-pretrained,
and any ensemble_summary.json files), and assembles ONE comparison table
(approach x region -> val/test accuracy) as both a CSV and a printed
Markdown table -- ready to paste into the paper's results section.

Usage:
    python3 collect_results_table.py --results-dir ~/mediinsight_project/results \
        --output-csv tier2_full_comparison.csv
"""

import argparse
import glob
import json
import os

import pandas as pd


def label_from_dirname(dirname: str) -> str:
    """Best-effort human-readable approach label from a results-dir name."""
    name = os.path.basename(dirname.rstrip("/"))
    # strip the "tier2_<REGION>_" prefix, keep the rest as the approach label
    for region in ("XR_SHOULDER", "XR_ELBOW", "XR_HAND"):
        prefix = f"tier2_{region}_"
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def region_from_dirname(dirname: str) -> str:
    name = os.path.basename(dirname.rstrip("/"))
    for region in ("XR_SHOULDER", "XR_ELBOW", "XR_HAND"):
        if f"tier2_{region}_" in name:
            return region
    return "UNKNOWN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--output-csv", default="tier2_full_comparison.csv")
    args = ap.parse_args()

    rows = []

    # Every per-model summary.json
    for summary_path in sorted(glob.glob(os.path.join(args.results_dir, "tier2_*", "summary.json"))):
        d = os.path.dirname(summary_path)
        with open(summary_path) as f:
            s = json.load(f)
        rows.append({
            "region": s.get("region", region_from_dirname(d)),
            "approach": label_from_dirname(d),
            "model_detail": s.get("model") or s.get("backbone") or "",
            "val_accuracy": s.get("val_accuracy"),
            "test_accuracy": s.get("test_accuracy"),
            "source_dir": d,
        })

    # Every ensemble_summary.json (reports its own set of rows per region)
    for ens_path in sorted(glob.glob(os.path.join(args.results_dir, "tier2_*_ensemble", "ensemble_summary.json"))):
        d = os.path.dirname(ens_path)
        with open(ens_path) as f:
            e = json.load(f)
        region = e.get("region", region_from_dirname(d))
        for name, acc in e.get("individual_test_accuracy", {}).items():
            rows.append({
                "region": region, "approach": f"(ensembled-input) {name}",
                "model_detail": name, "val_accuracy": None, "test_accuracy": acc,
                "source_dir": d,
            })
        rows.append({
            "region": region, "approach": "ensemble_simple_average",
            "model_detail": "", "val_accuracy": None,
            "test_accuracy": e.get("simple_average_test_accuracy"), "source_dir": d,
        })
        rows.append({
            "region": region, "approach": "ensemble_stacked_logreg",
            "model_detail": "", "val_accuracy": None,
            "test_accuracy": e.get("stacked_logreg_test_accuracy"), "source_dir": d,
        })

    if not rows:
        print(f"No summary.json or ensemble_summary.json files found under {args.results_dir} -- "
              f"nothing to collect yet.")
        return

    df = pd.DataFrame(rows).sort_values(["region", "test_accuracy"], ascending=[True, False])
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.output_csv}\n")

    for region in sorted(df["region"].unique()):
        sub = df[df["region"] == region][["approach", "model_detail", "val_accuracy", "test_accuracy"]]
        print(f"\n===== {region} (sorted by test accuracy, best first) =====")
        try:
            print(sub.to_markdown(index=False, floatfmt=".4f"))
        except ImportError:
            # `tabulate` (to_markdown's dependency) may not be installed --
            # plain to_string still gives a readable table either way.
            print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
