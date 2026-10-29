#!/bin/bash
for S in 1 2 3; do
  for R in shoulder elbow hand; do
    T=run_fm_dino_s1_${R}.pbs
    [ -f "$T" ] || { echo "missing template $T"; exit 1; }
    sed "s/fm_dino_s1_/fm_large_s${S}_/g; s#hf_cache/dinov2-base#hf_cache/dinov2-large#; s/_dinov2_base_s1/_dinov2_large_s${S}/; s/--seed 1/--seed ${S}/" $T > run_fm_large_s${S}_${R}.pbs
    sed "s/fm_dino_s1_/fm_lr1e5_s${S}_/g; s/_dinov2_base_s1/_dinov2_base_lr1e5_s${S}/; s/--seed 1 --tta/--seed ${S} --tta --lr-ft 1e-5 --patience 12/" $T > run_fm_lr1e5_s${S}_${R}.pbs
    sed "s/fm_dino_s1_/fm_rad_s${S}_/g; s#hf_cache/dinov2-base#hf_cache/rad-dino#; s/_dinov2_base_s1/_rad_dino_s${S}/; s/--seed 1/--seed ${S}/" $T > run_fm_rad_s${S}_${R}.pbs
  done
done
chmod +x run_fm_large_s*.pbs run_fm_lr1e5_s*.pbs run_fm_rad_s*.pbs
echo "Generated: $(ls run_fm_large_s*.pbs | wc -l) large + $(ls run_fm_lr1e5_s*.pbs | wc -l) lr1e5 + $(ls run_fm_rad_s*.pbs | wc -l) rad"
