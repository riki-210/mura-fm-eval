#!/bin/bash
# Run on the cluster login node, in ~/mediinsight_project/codes, with the fm env active.
set -e
cd ~/mediinsight_project/codes
grep -q -- "--exclude-csv" train_fm_dinov2.py || python3 patch_fm_exclude.py      # adds --exclude-csv (train/val only)
ls ~/mediinsight_project/results/label_cleaning/exclude_XR_*.csv                 # must list three files
bash gen_cv_jobs.sh                  # writes run_clean_s{1,2,3}_{elbow,hand,shoulder}.pbs (flagged studies removed)
bash gen_randexcl_jobs.sh            # writes run_rand_s{1,2,3}_{elbow,hand,shoulder}.pbs (random removal, same count)
for f in run_clean_s?_*.pbs run_rand_s?_*.pbs; do qsub "$f"; done
qstat -u $USER
