import glob
BLOCK = r'''{
export TF_FORCE_GPU_ALLOW_GROWTH=true
sleep $((RANDOM % 90))
echo "CUDA_VISIBLE_DEVICES(before)=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
fi
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
'''
n = 0
for f in sorted(glob.glob("run_round3_320_s*_*.pbs")):
    s = open(f).read()
    if "TF_FORCE_GPU_ALLOW_GROWTH" in s:
        continue
    assert s.count("\n{\n") == 1, f
    open(f, "w").write(s.replace("\n{\n", "\n" + BLOCK, 1)); n += 1
print("patched", n, "files")
