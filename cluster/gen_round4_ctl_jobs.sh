#!/bin/bash
for SEED in 1 2 3; do
  for R in shoulder elbow hand; do
    REG=XR_$(echo $R | tr 'a-z' 'A-Z')
    N="r4_224_s${SEED}_${R}"
    cat > run_round4_224_s${SEED}_${R}.pbs <<EOF
#!/bin/bash
#PBS -N ${N}
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o ${N}.pbsout

set -e
cd "\$PBS_O_WORKDIR" 2>/dev/null || cd ~/mediinsight_project/codes

source ~/miniforge3/etc/profile.d/conda.sh
conda activate mediinsight
LIVELOG="${N}_live.log"
W="\$HOME/mediinsight_project/codes/graz_pretrained_DenseNet169.weights.h5"
{
export TF_FORCE_GPU_ALLOW_GROWTH=true
sleep \$((RANDOM % 90))
echo "CUDA_VISIBLE_DEVICES(before)=\$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
if [ -z "\$CUDA_VISIBLE_DEVICES" ]; then
  HN=\$(hostname -s); mkdir -p "\$HOME/.r3_locks"
  while true; do
    for i in \$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | cut -d, -f1); do
      used=\$(nvidia-smi -i \$i --query-gpu=memory.used --format=csv,noheader,nounits)
      if [ "\$used" -lt 55000 ] && mkdir "\$HOME/.r3_locks/\${HN}_\$i" 2>/dev/null; then export CUDA_VISIBLE_DEVICES=\$i; break 2; fi
    done
    echo "no GPU with >=27GB free, waiting 60s..."; sleep 60
  done
  trap 'rmdir "\$HOME/.r3_locks/\${HN}_\$CUDA_VISIBLE_DEVICES" 2>/dev/null' EXIT
fi
echo "Using GPU: \$CUDA_VISIBLE_DEVICES"
echo "===== JOB INFO ====="; echo "Job ID: \$PBS_JOBID"; echo "Node: \$(hostname)"; echo "Start: \$(date)"; echo "====================="
if [ ! -f "\$W" ]; then echo "ERROR: \$W not found" >&2; exit 1; fi
python3 -u train_tier2_strong.py \\
    --index-csv "\$HOME/mediinsight_project/data/mura_ehr_index.csv" \\
    --region ${REG} \\
    --output-dir "\$HOME/mediinsight_project/results/tier2_${REG}_graz_d169_224_s${SEED}" \\
    --backbone DenseNet169 --domain-weights "\$W" \\
    --img-size 224 --seed ${SEED} --tta \\
    --dropout 0.5 --l2 1e-4 --warmup-epochs 5 --finetune-epochs 60 --save-probs
echo "===== JOB DONE ====="; echo "End: \$(date)"
} > "\$LIVELOG" 2>&1
EOF
  done
done
cat > submit_round4.sh <<'EOF'
#!/bin/bash
cd ~/mediinsight_project/codes || exit 1
for f in run_round4_224_s*_*.pbs; do echo "$f -> $(qsub $f)"; done
qstat -u $USER
EOF
chmod +x run_round4_224_s*_*.pbs submit_round4.sh
echo "Generated: $(ls run_round4_224_s*_*.pbs | wc -l) PBS files + submit_round4.sh"
