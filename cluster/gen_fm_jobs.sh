#!/bin/bash
hdr() {  # $1 = job name  $2 = walltime
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
conda activate fm
export HF_HUB_OFFLINE=1
LIVELOG="$1_live.log"
RES="\$HOME/mediinsight_project/results"
{
sleep \$((RANDOM % 90))
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
echo "Using GPU: \$CUDA_VISIBLE_DEVICES"; echo "Start: \$(date)"
EOF
}
IDX='"$HOME/mediinsight_project/data/mura_ehr_index.csv"'
MD='"$HOME/hf_cache/dinov2-base"'

{ hdr fm_smoke 00:30:00
cat <<EOF
python3 -u train_fm_dinov2.py --index-csv $IDX --region XR_ELBOW --model-dir $MD \\
    --output-dir "\$RES/fm_smoke" --limit 128 --warmup-epochs 1 --finetune-epochs 2 --seed 1 --tta
echo "===== JOB DONE ====="; echo "End: \$(date)"
} > "\$LIVELOG" 2>&1
EOF
} > run_fm_smoke.pbs

for R in shoulder elbow hand; do
  REG=XR_$(echo $R | tr 'a-z' 'A-Z')
  { hdr fm_dino_s1_${R} 24:00:00
  cat <<EOF
python3 -u train_fm_dinov2.py --index-csv $IDX --region ${REG} --model-dir $MD \\
    --output-dir "\$RES/tier2_${REG}_dinov2_base_s1" --seed 1 --tta
echo "===== JOB DONE ====="; echo "End: \$(date)"
} > "\$LIVELOG" 2>&1
EOF
  } > run_fm_dino_s1_${R}.pbs
done
chmod +x run_fm_*.pbs
echo "Generated: $(ls run_fm_*.pbs | wc -l) PBS files (1 smoke + 3 real)"
