#!/bin/bash
for SEED in 1 2 3; do
  for R in shoulder elbow hand; do
    REG=XR_$(echo $R | tr 'a-z' 'A-Z')
    N="r3_320_s${SEED}_${R}"
    cat > run_round3_320_s${SEED}_${R}.pbs <<EOF
#!/bin/bash
#PBS -N ${N}
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o ${N}.pbsout

set -e
cd "\$PBS_O_WORKDIR" 2>/dev/null || cd ~/mediinsight_project/codes

source ~/miniforge3/etc/profile.d/conda.sh
conda activate mediinsight
LIVELOG="${N}_live.log"
W="\$HOME/mediinsight_project/codes/graz_pretrained_DenseNet169.weights.h5"
{
echo "===== JOB INFO ====="; echo "Job ID: \$PBS_JOBID"; echo "Node: \$(hostname)"; echo "Start: \$(date)"; echo "====================="
if [ ! -f "\$W" ]; then echo "ERROR: \$W not found" >&2; exit 1; fi
python3 -u train_tier2_strong.py \\
    --index-csv "\$HOME/mediinsight_project/data/mura_ehr_index.csv" \\
    --region ${REG} \\
    --output-dir "\$HOME/mediinsight_project/results/tier2_${REG}_graz_d169_320_s${SEED}" \\
    --backbone DenseNet169 --domain-weights "\$W" \\
    --img-size 320 --seed ${SEED} --tta \\
    --dropout 0.5 --l2 1e-4 --warmup-epochs 5 --finetune-epochs 60 --save-probs
echo "===== JOB DONE ====="; echo "End: \$(date)"
} > "\$LIVELOG" 2>&1
EOF
  done
done
cat > submit_round3.sh <<'EOF'
#!/bin/bash
cd ~/mediinsight_project/codes || exit 1
for f in run_round3_320_s*_*.pbs; do echo "$f -> $(qsub $f)"; done
qstat -u $USER
EOF
chmod +x run_round3_320_s*_*.pbs submit_round3.sh
echo "Generated: $(ls run_round3_320_s*_*.pbs | wc -l) PBS files + submit_round3.sh"
