#!/bin/bash
# Makes run_erase_smoke.pbs and run_erase_s<S>_<region>.pbs (DINOv2-base image-level + random-erasing augmentation)
cd ~/mediinsight_project/codes
python3 - <<'PY'
import re
IDX = '"$HOME/mediinsight_project/data/mura_ehr_index.csv"'
MD = '"$HOME/hf_cache/dinov2-base"'
for reg in ("elbow", "hand", "shoulder"):
    R = "XR_" + reg.upper()
    t = open(f"run_fm_dino_s1_{reg}.pbs").read()
    m = re.search(r"python3 -u train_fm_dinov2\.py.*?--tta\n", t, re.S)
    assert m, "template python line not found"
    for seed in (1, 2, 3):
        cmd = (f'python3 -u train_fm_dinov2.py --index-csv {IDX} --region {R} --model-dir {MD} '
               f'--output-dir "$RES/tier2_{R}_dinov2_base_erase_s{seed}" --erase-p 0.5 --seed {seed} --tta\n')
        open(f"run_erase_s{seed}_{reg}.pbs", "w").write(t.replace(m.group(0), cmd).replace(f"fm_dino_s1_{reg}", f"erase_s{seed}_{reg}"))
t = open("run_fm_dino_s1_hand.pbs").read()
m = re.search(r"python3 -u train_fm_dinov2\.py.*?--tta\n", t, re.S)
cmd = (f'python3 -u train_fm_dinov2.py --index-csv {IDX} --region XR_HAND --model-dir {MD} --output-dir "$RES/erase_smoke" '
       f'--erase-p 0.5 --limit 128 --warmup-epochs 1 --finetune-epochs 2 --tta\n')
s = t.replace(m.group(0), cmd).replace("fm_dino_s1_hand", "erase_smoke").replace("walltime=24:00:00", "walltime=00:30:00")
open("run_erase_smoke.pbs", "w").write(s)
print("generated: run_erase_smoke.pbs, run_erase_s{1,2,3}_{elbow,hand,shoulder}.pbs")
PY
