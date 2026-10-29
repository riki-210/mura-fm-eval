import glob, re
NEW = '''if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  HN=$(hostname -s); mkdir -p "$HOME/.r3_locks"
  while true; do
    for i in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | cut -d, -f1); do
      used=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)
      if [ "$used" -lt 55000 ] && mkdir "$HOME/.r3_locks/${HN}_$i" 2>/dev/null; then export CUDA_VISIBLE_DEVICES=$i; break 2; fi
    done
    echo "no GPU with >=27GB free, waiting 60s..."; sleep 60
  done
  trap 'rmdir "$HOME/.r3_locks/${HN}_$CUDA_VISIBLE_DEVICES" 2>/dev/null' EXIT
fi
'''
n = 0
for f in sorted(glob.glob("run_round3_320_s*_*.pbs")):
    s = open(f).read()
    if ".r3_locks" in s:
        continue
    s2, k = re.subn(r'if \[ -z "\$CUDA_VISIBLE_DEVICES" \]; then\n.*?\nfi\n', lambda m: NEW, s, count=1, flags=re.S)
    assert k == 1, f
    open(f, "w").write(s2); n += 1
print("patched", n, "files")
