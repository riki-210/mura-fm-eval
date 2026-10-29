# mura-fm-eval

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23761625.svg)](https://doi.org/10.5281/zenodo.23761625)

Code, partitions and per-run predictions for:

> Sarkar, A. and Soni, D. *Foundation-encoder transfer, study-level learning and label noise in abnormality
> detection on musculoskeletal radiographs: a leakage-free, multi-seed evaluation.*
> Jaypee Institute of Information Technology, Noida.

Everything reported in the paper (accuracy, bootstrap intervals, paired tests, Holm/BH correction,
calibration, operating points, label audit, retraining without flagged studies) is computed from the stored
test predictions in `predictions/`, so the tables and figures can be regenerated without retraining.

## Layout

| Folder | Content |
| --- | --- |
| `data/` | `test_index.csv`: test images with patient id, region, label and split (paths relative to `MURA-v1.1/`). `data/partitions/` holds the full MURA index (`mura_ehr_index.csv`), the pooled index with cross-validation folds (`pooled_index.csv`) and the GRAZPEDWRI-DX index; `data/label_cleaning/` holds the studies flagged by the audit (`label_issues_*`, `exclude_*`) and the random-removal control (`randexcl_*`). |
| `predictions/` | One folder per run (`tier2_<REGION>_<model>_s<seed>`): `test_probs.npy`, `val_probs.npy`, labels, `summary.json`. Model weights are not included. |
| `analysis/` | Study-level evaluation: `load.py` (loader), `stats.py` (paired bootstrap), `corr.py` (Holm/BH), `more.py` (ROC, calibration, operating points, overlap), `clean.py` (retraining without flagged studies), `diag.py` (method diagrams). |
| `cluster/` | PBS job generators (`gen_*_jobs.sh`) and the job scripts (`run_*.pbs`) used for every experiment, including the label-cleaning and random-removal runs. |
| `training/` | Training and data-preparation scripts: DINOv2 image-level (`train_fm_dinov2.py`), study-level MIL (`train_fm_mil.py`), all-body-parts (`train_fm_all7.py`), cross-validation audit (`train_fm_cv.py`, `find_label_issues.py`), GRAZ-pretrained CNNs and YOLO11m. |
| `results/`, `figures/` | JSON outputs and figure PDFs used in the paper. |

## Reproduce the analysis

```bash
pip install -r requirements.txt
cd analysis
export PRED_DIR=../predictions TEST_INDEX=../data/test_index.csv   # defaults
python stats.py     # paired bootstrap comparisons  -> comps.json
python corr.py      # Holm / Benjamini-Hochberg      -> corr.json
python more.py      # ROC, calibration, operating points, overlap
python clean.py     # flagged vs random removal vs baseline (Section 4.16)
```

Expected output of `clean.py` (pooled, study level, threshold 0.5): DINOv2-base 82.67, flagged removed 83.58,
random removed 82.83; flagged vs base +0.91 (95% CI -0.03 to 1.82, p = 0.060).

## Data

The MURA and GRAZPEDWRI-DX images are not redistributed. Obtain them from their owners and respect their licences:
MURA (Stanford ML Group), GRAZPEDWRI-DX (Nagy et al., 2022, Scientific Data 9, 222). Image paths in the index files are
relative to the dataset folder.

## Notes

* Partitions are at patient level, seeds are 1, 2 and 3, and thresholds are chosen on validation data only.
* Parts of the code, statistical analysis, figures and manuscript were prepared with the assistance of Claude
  (Anthropic); the authors reviewed all output.

## Citation and licence

Code: MIT licence (see `LICENSE`). Please cite the paper; archived on Zenodo: https://doi.org/10.5281/zenodo.23761625 .
