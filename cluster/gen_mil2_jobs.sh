#!/bin/bash
# Makes run_mil2_smoke.pbs and run_mil2_s<S>_<region>.pbs (warm start from the image-level DINOv2 of the same seed)
cd ~/mediinsight_project/codes
python3 - <<'PY'
import re
IDX = '"$HOME/mediinsight_project/data/mura_ehr_index.csv"'
MD = '"$HOME/hf_cache/dinov2-base"'
OPT = "--accum 4 --lr-head 5e-4 --lr-ft 1e-5"
for reg in ("elbow", "hand", "shoulder"):
    R = "XR_" + reg.upper()
    t = open(f"run_fm_dino_s1_{reg}.pbs").read()
    m = re.search(r"python3 -u train_fm_dinov2\.py.*?--tta\n", t, re.S)
    assert m, "template python line not found"
    for seed in (1, 2, 3):
        cmd = (f'python3 -u train_fm_mil.py --index-csv {IDX} --region {R} --model-dir {MD} '
               f'--init-from "$RES/tier2_{R}_dinov2_base_s{seed}/best_model.pt" {OPT} '
               f'--output-dir "$RES/tier2_{R}_dinov2_mil2_s{seed}" --seed {seed} --tta\n')
        open(f"run_mil2_s{seed}_{reg}.pbs", "w").write(t.replace(m.group(0), cmd).replace(f"fm_dino_s1_{reg}", f"mil2_s{seed}_{reg}"))
t = open("run_fm_dino_s1_hand.pbs").read()
m = re.search(r"python3 -u train_fm_dinov2\.py.*?--tta\n", t, re.S)
cmd = (f'python3 -u train_fm_mil.py --index-csv {IDX} --region XR_HAND --model-dir {MD} '
       f'--init-from "$RES/tier2_XR_HAND_dinov2_base_s1/best_model.pt" {OPT} --output-dir "$RES/mil2_smoke" '
       f'--limit 64 --warmup-epochs 1 --finetune-epochs 2 --tta\n')
s = t.replace(m.group(0), cmd).replace("fm_dino_s1_hand", "mil2_smoke").replace("walltime=24:00:00", "walltime=00:30:00")
open("run_mil2_smoke.pbs", "w").write(s)
print("generated: run_mil2_smoke.pbs, run_mil2_s{1,2,3}_{elbow,hand,shoulder}.pbs")
PY
