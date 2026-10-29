#!/bin/bash
# Makes run_all7_smoke.pbs and run_all7_s{1,2,3}.pbs (one job per seed; each job writes all 3 target regions)
cd ~/mediinsight_project/codes
python3 - <<'PY'
import re
IDX = '"$HOME/mediinsight_project/data/mura_ehr_index.csv"'
EXTRA = '"$HOME/mediinsight_project/data/mura_extra_regions_train.csv"'
MD = '"$HOME/hf_cache/dinov2-base"'
t = open("run_fm_dino_s1_hand.pbs").read()
m = re.search(r"python3 -u train_fm_dinov2\.py.*?--tta\n", t, re.S)
assert m, "template python line not found"
for seed in (1, 2, 3):
    cmd = (f'python3 -u train_fm_all7.py --index-csv {IDX} --extra-train-csv {EXTRA} --model-dir {MD} '
           f'--results-dir "$RES" --tag all7 --seed {seed} --tta\n')
    open(f"run_all7_s{seed}.pbs", "w").write(t.replace(m.group(0), cmd).replace("fm_dino_s1_hand", f"all7_s{seed}"))
cmd = (f'python3 -u train_fm_all7.py --index-csv {IDX} --extra-train-csv {EXTRA} --model-dir {MD} '
       f'--results-dir "$RES/all7_smoke" --tag all7smoke --limit 64 --warmup-epochs 1 --finetune-epochs 2 --tta\n')
s = t.replace(m.group(0), cmd).replace("fm_dino_s1_hand", "all7_smoke").replace("walltime=24:00:00", "walltime=00:30:00")
open("run_all7_smoke.pbs", "w").write(s)
print("generated: run_all7_smoke.pbs, run_all7_s{1,2,3}.pbs")
PY
