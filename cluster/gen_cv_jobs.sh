#!/bin/bash
# Makes run_cv_<region>.pbs (5-fold OOF) and run_clean_s<S>_<region>.pbs (retrain DINOv2-base on cleaned train/val)
cd ~/mediinsight_project/codes
python3 - <<'PY'
import re
for reg in ("elbow", "hand", "shoulder"):
    R = "XR_" + reg.upper()
    t = open(f"run_fm_dino_s1_{reg}.pbs").read()
    m = re.search(r"python3 -u train_fm_dinov2\.py.*?--tta\n", t, re.S)
    assert m, "template python line not found"
    cv = ('python3 -u train_fm_cv.py --index-csv "$HOME/mediinsight_project/data/mura_ehr_index.csv" --region %s '
          '--model-dir "$HOME/hf_cache/dinov2-base" --output-dir "$RES/cv_%s"\n' % (R, R))
    s = t.replace(m.group(0), cv).replace(f"fm_dino_s1_{reg}", f"cv_{reg}")
    open(f"run_cv_{reg}.pbs", "w").write(s)
    for seed in (1, 2, 3):
        cl = ('python3 -u train_fm_dinov2.py --index-csv "$HOME/mediinsight_project/data/mura_ehr_index.csv" --region %s '
              '--model-dir "$HOME/hf_cache/dinov2-base" --output-dir "$RES/tier2_%s_dinov2_base_clean_s%d" '
              '--exclude-csv "$HOME/mediinsight_project/results/label_cleaning/exclude_%s.csv" --seed %d --tta\n' % (R, R, seed, R, seed))
        s = t.replace(m.group(0), cl).replace(f"fm_dino_s1_{reg}", f"clean_s{seed}_{reg}")
        open(f"run_clean_s{seed}_{reg}.pbs", "w").write(s)
print("generated: run_cv_{elbow,hand,shoulder}.pbs, run_clean_s{1,2,3}_{elbow,hand,shoulder}.pbs")
PY
