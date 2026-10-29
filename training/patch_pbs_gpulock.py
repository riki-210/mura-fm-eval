import glob, re
NEW = '''if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  while true; do
    for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
      used=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)
      if [ "$used" -lt 3000 ] && mkdir /tmp/r3_gpulock_$i 2>/dev/null; then export CUDA_VISIBLE_DEVICES=$i; break 2; fi
    done
    echo "no free GPU, waiting 60s..."; sleep 60
  done
  trap 'rmdir /tmp/r3_gpulock_$CUDA_VISIBLE_DEVICES 2>/dev/null' EXIT
fi
'''
n = 0
for f in sorted(glob.glob("run_round3_320_s*_*.pbs")):
    s = open(f).read()
    if "r3_gpulock_" in s:
        continue
    s2, k = re.subn(r'if \[ -z "\$CUDA_VISIBLE_DEVICES" \]; then\n.*?\nfi\n', lambda m: NEW, s, count=1, flags=re.S)
    assert k == 1, f
    open(f, "w").write(s2); n += 1
print("patched", n, "files")
