#!/bin/bash
for R in shoulder elbow hand; do
  REG=XR_$(echo $R | tr 'a-z' 'A-Z')
  N="gate_crop_${R}"
  cat > run_gate_crop_${R}.pbs <<EOF
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
RES="\$HOME/mediinsight_project/results"
{
export TF_FORCE_GPU_ALLOW_GROWTH=true
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
MP="\$RES/tier2_${REG}_graz_pretrained_densenet169/best_model.h5"
if [ ! -f "\$MP" ]; then echo "ERROR: \$MP not found" >&2; exit 1; fi
python3 -u attention_boxes.py --index-csv "\$HOME/mediinsight_project/data/mura_ehr_index.csv" \\
    --region ${REG} --model-path "\$MP" --out-npz "\$RES/crop_boxes_${REG}.npz" \\
    --img-size 224 --frac 0.6 --montage-png "\$RES/crop_check_${REG}.png"
python3 -u train_crop_gate.py --boxes "\$RES/crop_boxes_${REG}.npz" --hi-res 448 \\
    --index-csv "\$HOME/mediinsight_project/data/mura_ehr_index.csv" \\
    --region ${REG} --output-dir "\$RES/tier2_${REG}_graz_d169_crop60_s1" \\
    --backbone DenseNet169 --domain-weights "\$W" \\
    --img-size 224 --seed 1 --tta \\
    --dropout 0.5 --l2 1e-4 --warmup-epochs 5 --finetune-epochs 60 --save-probs
echo "===== JOB DONE ====="; echo "End: \$(date)"
} > "\$LIVELOG" 2>&1
EOF
done
chmod +x run_gate_crop_*.pbs
echo "Generated: $(ls run_gate_crop_*.pbs | wc -l) PBS files"
