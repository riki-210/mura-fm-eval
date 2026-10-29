"""
ensemble_stack_tier2.py -- combines predictions from several already-trained
Tier 2 models (any mix of train_tier2_strong.py / train_tier2_vit.py /
train_tier2_hybrid.py runs, as long as each was run with --save-probs) into
one final stacked ensemble, per region, using a logistic-regression
meta-classifier trained on the validation-set probabilities -- the same
late-fusion/stacking approach already proven in Paper 1
(clinical RF + MRI ensemble votes combined by a logistic-regression
meta-model).

Why stacking rather than simple probability averaging: a learned
meta-classifier can down-weight a model that's confidently wrong in a
particular way, rather than treating every model's vote equally regardless
of its individual reliability -- generally a few points better than a flat
average when the constituent models have noticeably different accuracy
levels (which is expected here: AlexNet-family vs. DenseNet/ViT/Hybrid).

Usage:
    python3 ensemble_stack_tier2.py --region XR_SHOULDER \
        --model-dir results/tier2_XR_SHOULDER_strong_densenet169 \
        --model-dir results/tier2_XR_SHOULDER_vit_b16 \
        --model-dir results/tier2_XR_SHOULDER_hybrid_densenet169_vitb16 \
        --model-dir results/tier2_XR_SHOULDER_pretrained \
        --output-dir results/tier2_XR_SHOULDER_ensemble

Each --model-dir must contain val_probs.npy, test_probs.npy, val_labels.npy,
test_labels.npy (written automatically when a training script is run with
--save-probs). Labels across all given model-dirs for the same region must
line up row-for-row (true as long as each script was run against the SAME
mura_ehr_index.csv -- the split is deterministic/seeded, so this holds).
"""

import argparse
import os
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--model-dir", action="append", required=True,
                     help="Directory containing val_probs.npy/test_probs.npy/val_labels.npy/"
                          "test_labels.npy for one trained model. Repeat for each model to ensemble.")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_names = []
    val_prob_list, test_prob_list = [], []
    val_labels, test_labels = None, None

    for d in args.model_dir:
        summary_path = os.path.join(d, "summary.json")
        name = d
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                s = json.load(f)
            name = s.get("model") or s.get("backbone") or d
        model_names.append(name)

        vp = np.load(os.path.join(d, "val_probs.npy"))
        tp = np.load(os.path.join(d, "test_probs.npy"))
        vl = np.load(os.path.join(d, "val_labels.npy"))
        tl = np.load(os.path.join(d, "test_labels.npy"))

        if val_labels is None:
            val_labels, test_labels = vl, tl
        else:
            assert np.array_equal(val_labels, vl) and np.array_equal(test_labels, tl), (
                f"Label mismatch in {d} -- did every model train on the same mura_ehr_index.csv "
                f"and region split? Ensembling requires identical val/test row order across all models."
            )

        val_prob_list.append(vp)
        test_prob_list.append(tp)
        print(f"Loaded {name}: val_probs {vp.shape}, test_probs {tp.shape}")

    # Stack each model's class-1 (fracture) probability as one feature column
    # for the meta-classifier -- simple, standard stacking-ensemble input.
    X_val = np.column_stack([p[:, 1] for p in val_prob_list])
    X_test = np.column_stack([p[:, 1] for p in test_prob_list])

    # Individual-model baselines (for the comparison table)
    individual_results = {}
    for name, tp in zip(model_names, test_prob_list):
        pred = np.argmax(tp, axis=1)
        individual_results[name] = float(accuracy_score(test_labels, pred))

    # Simple averaging baseline (unweighted ensemble, no learning)
    avg_test_probs = np.mean(np.stack(test_prob_list, axis=0), axis=0)
    avg_pred = np.argmax(avg_test_probs, axis=1)
    avg_acc = float(accuracy_score(test_labels, avg_pred))

    # Stacked logistic-regression meta-classifier (learned on val set, evaluated on test)
    meta = LogisticRegression(max_iter=1000)
    meta.fit(X_val, val_labels)
    stacked_pred = meta.predict(X_test)
    stacked_acc = float(accuracy_score(test_labels, stacked_pred))
    stacked_report = classification_report(test_labels, stacked_pred, output_dict=True)

    result = {
        "region": args.region,
        "models_ensembled": model_names,
        "individual_test_accuracy": individual_results,
        "simple_average_test_accuracy": avg_acc,
        "stacked_logreg_test_accuracy": stacked_acc,
        "stacked_meta_coefficients": dict(zip(model_names, meta.coef_[0].tolist())),
        "stacked_classification_report": stacked_report,
    }
    with open(os.path.join(args.output_dir, "ensemble_summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n===== Region: {args.region} =====")
    print("Individual model test accuracies:")
    for name, acc in individual_results.items():
        print(f"  {name}: {acc:.4f}")
    print(f"Simple average ensemble test accuracy: {avg_acc:.4f}")
    print(f"Stacked logistic-regression ensemble test accuracy: {stacked_acc:.4f}")
    print(f"Meta-classifier coefficients (higher = more weight given to that model): "
          f"{result['stacked_meta_coefficients']}")


if __name__ == "__main__":
    main()
