#!/bin/bash
HEADER() { # name walltime
cat <<EOF
#!/bin/bash
#PBS -N $1
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=$2
#PBS -j oe
#PBS -o $1.pbsout

set -e
cd "\$PBS_O_WORKDIR" 2>/dev/null || cd ~/mediinsight_project/codes

source ~/miniforge3/etc/profile.d/conda.sh
conda activate mediinsight
EOF
}

for BB in DenseNet121 EfficientNetV2B0; do
  lc=$(echo "$BB" | tr 'A-Z' 'a-z')
  N="graz_pre_${lc}"
  { HEADER "$N" 48:00:00
cat <<EOF
LIVELOG="${N}_live.log"
{
echo "===== JOB INFO ====="; echo "Job ID: \$PBS_JOBID"; echo "Node: \$(hostname)"; echo "Start: \$(date)"; echo "====================="
python3 -u pretrain_graz_backbone.py \\
    --index-csv "\$HOME/mediinsight_project/data/graz_index.csv" \\
    --output-weights "\$HOME/mediinsight_project/codes/graz_pretrained_${BB}.weights.h5" \\
    --backbone ${BB} --epochs 30
echo "===== JOB DONE ====="; echo "End: \$(date)"
} > "\$LIVELOG" 2>&1
EOF
  } > run_graz_pretrain_${lc}.pbs
done

N="graz_pre_vit"
{ HEADER "$N" 48:00:00
cat <<'EOF'
LIVELOG="graz_pre_vit_live.log"
{
echo "===== JOB INFO ====="; echo "Job ID: $PBS_JOBID"; echo "Node: $(hostname)"; echo "Start: $(date)"; echo "====================="
python3 -u pretrain_graz_vit.py \
    --index-csv "$HOME/mediinsight_project/data/graz_index.csv" \
    --output-weights "$HOME/mediinsight_project/codes/graz_pretrained_vit_b16.weights.h5"
echo "===== JOB DONE ====="; echo "End: $(date)"
} > "$LIVELOG" 2>&1
EOF
} > run_graz_pretrain_vit.pbs

for BB in DenseNet121 EfficientNetV2B0; do
  lc=$(echo "$BB" | tr 'A-Z' 'a-z')
  for R in shoulder elbow hand; do
    REG=XR_$(echo $R | tr 'a-z' 'A-Z')
    N="graz_ft_${lc}_${R}"
    { HEADER "$N" 24:00:00
cat <<EOF
LIVELOG="${N}_live.log"
W="\$HOME/mediinsight_project/codes/graz_pretrained_${BB}.weights.h5"
{
echo "===== JOB INFO ====="; echo "Job ID: \$PBS_JOBID"; echo "Node: \$(hostname)"; echo "Start: \$(date)"; echo "====================="
if [ ! -f "\$W" ]; then echo "ERROR: \$W not found -- GRAZ pretraining must finish first." >&2; exit 1; fi
python3 -u train_tier2_strong.py \\
    --index-csv "\$HOME/mediinsight_project/data/mura_ehr_index.csv" \\
    --region ${REG} \\
    --output-dir "\$HOME/mediinsight_project/results/tier2_${REG}_graz_pretrained_${lc}" \\
    --backbone ${BB} --domain-weights "\$W" \\
    --dropout 0.5 --l2 1e-4 --warmup-epochs 5 --finetune-epochs 60 --save-probs
echo "===== JOB DONE ====="; echo "End: \$(date)"
} > "\$LIVELOG" 2>&1
EOF
    } > run_tier2_graz_${lc}_${R}.pbs
  done
done

for R in shoulder elbow hand; do
  REG=XR_$(echo $R | tr 'a-z' 'A-Z')
  N="graz_ft_vit_${R}"
  { HEADER "$N" 24:00:00
cat <<EOF
LIVELOG="${N}_live.log"
W="\$HOME/mediinsight_project/codes/graz_pretrained_vit_b16.weights.h5"
{
echo "===== JOB INFO ====="; echo "Job ID: \$PBS_JOBID"; echo "Node: \$(hostname)"; echo "Start: \$(date)"; echo "====================="
if [ ! -f "\$W" ]; then echo "ERROR: \$W not found -- GRAZ ViT pretraining must finish first." >&2; exit 1; fi
python3 -u train_tier2_vit.py \\
    --index-csv "\$HOME/mediinsight_project/data/mura_ehr_index.csv" \\
    --region ${REG} \\
    --output-dir "\$HOME/mediinsight_project/results/tier2_${REG}_graz_pretrained_vit_b16" \\
    --domain-weights "\$W" \\
    --dropout 0.5 --l2 1e-4 --warmup-epochs 5 --finetune-epochs 60 --save-probs
echo "===== JOB DONE ====="; echo "End: \$(date)"
} > "\$LIVELOG" 2>&1
EOF
  } > run_tier2_graz_vit_${R}.pbs
done

cat > submit_graz_chain.sh <<'EOF'
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
EOF
chmod +x run_graz_pretrain_*.pbs run_tier2_graz_*_*.pbs submit_graz_chain.sh
echo "Generated: $(ls run_graz_pretrain_*.pbs run_tier2_graz_*_*.pbs | wc -l) PBS files + submit_graz_chain.sh"
