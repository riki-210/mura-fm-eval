#!/bin/bash
# Makes run_rand_s{1,2,3}_{elbow,hand,shoulder}.pbs: DINOv2-base retrained with the same NUMBER of random studies removed (control).
cd ~/mediinsight_project/codes
python3 make_random_exclude.py || exit 1
python3 - <<'PY'
import re
for reg in ("elbow", "hand", "shoulder"):
    R = "XR_" + reg.upper()
    t = open(f"run_fm_dino_s1_{reg}.pbs").read()
    m = re.search(r"python3 -u train_fm_dinov2\.py.*?--tta\n", t, re.S)
    assert m, "template python line not found"
    for seed in (1, 2, 3):
        cl = ('python3 -u train_fm_dinov2.py --index-csv "$HOME/mediinsight_project/data/mura_ehr_index.csv" --region %s '
              '--model-dir "$HOME/hf_cache/dinov2-base" --output-dir "$RES/tier2_%s_dinov2_base_randexcl_s%d" '
              '--exclude-csv "$HOME/mediinsight_project/results/label_cleaning/randexcl_%s.csv" --seed %d --tta\n' % (R, R, seed, R, seed))
        open(f"run_rand_s{seed}_{reg}.pbs", "w").write(t.replace(m.group(0), cl).replace(f"fm_dino_s1_{reg}", f"rand_s{seed}_{reg}"))
print("generated run_rand_s{1,2,3}_{elbow,hand,shoulder}.pbs")
PY
