#!/bin/bash
cd ~/mediinsight_project/codes || exit 1
for BB in densenet121 efficientnetv2b0 vit; do
  PRE=$(qsub run_graz_pretrain_${BB}.pbs) || { echo "qsub failed for $BB pretrain"; exit 1; }
  echo "pretrain ${BB}: $PRE"
  for R in shoulder elbow hand; do
    FT=$(qsub -W depend=afterok:${PRE} run_tier2_graz_${BB}_${R}.pbs) || { echo "dependent qsub failed (${BB} ${R})"; exit 1; }
    echo "   fine-tune ${BB} ${R}: $FT (after $PRE)"
  done
done
qstat -u $USER
