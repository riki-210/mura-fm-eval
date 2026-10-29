#!/usr/bin/env bash
# Run on the cluster. Gathers training code, partition/label-audit files and job scripts into one folder,
# removes personal paths, and writes mura-fm-eval_cluster_files.zip. Model weights are never copied.
# Usage:  bash collect_repo.sh            (edit PROJECT if your project folder differs)
set -uo pipefail   # no -e: grep returns 1 when it finds nothing, which is fine here
PROJECT="${PROJECT:-$HOME/mediinsight_project}"
OUT="${OUT:-$HOME/repo_collect}"
rm -rf "$OUT"; mkdir -p "$OUT"/{training,cluster,data/partitions,data/label_cleaning}

# 1. training and job code (python, shell, pbs), no weights
find "$PROJECT/codes" -maxdepth 2 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.pbs' \) \
     ! -path '*/__pycache__/*' -exec cp {} "$OUT/training/" \;
# move job/pbs files into cluster/
mv "$OUT"/training/*.pbs "$OUT/cluster/" 2>/dev/null || true
mv "$OUT"/training/gen_*.sh "$OUT"/training/submit_*.sh "$OUT/cluster/" 2>/dev/null || true

# 2. partition indexes and label-audit lists (csv/txt/json), not images
find "$PROJECT" -maxdepth 3 -type f \( -iname '*index*.csv' -o -iname '*split*.csv' -o -iname '*partition*.csv' \) \
     ! -path '*/MURA-v1.1/*' ! -path '*/results/tier2_*' -exec cp {} "$OUT/data/partitions/" \; 2>/dev/null || true
cp "$PROJECT"/results/label_cleaning/*.csv "$OUT/data/label_cleaning/" 2>/dev/null || true

# 3. remove personal paths and usernames
USERNAME_PAT="$(whoami)"
grep -rlI -e "$HOME" -e "$USERNAME_PAT" "$OUT" 2>/dev/null | while read -r f; do
  sed -i -e "s#$HOME#~#g" -e "s#/home/[A-Za-z0-9._-]*/#~/#g" -e "s#$USERNAME_PAT#USER#g" "$f"
done
# index csv files: make image paths relative to the MURA folder
find "$OUT/data" -name '*.csv' -exec sed -i -E 's#(^|,)[^,]*/(MURA-v1\.1/)#\1\2#' {} \;

# 4. size and content check (should be small, no .pt files)
echo "Files collected:"; find "$OUT" -type f | sed "s#^$OUT/##" | sort
echo; echo "Weights found (must be none):"; find "$OUT" -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' | wc -l
echo "Remaining mentions of your home path (should be 0):"; grep -rIl "$HOME" "$OUT" 2>/dev/null | wc -l

# 5. per-run predictions (small) for any runs not already in the repository
mkdir -p "$OUT/predictions"
find "$PROJECT/results" -maxdepth 2 -type f \
  \( -name 'test_probs*.npy' -o -name 'val_probs*.npy' -o -name 'test_labels.npy' -o -name 'val_labels.npy' -o -name 'summary.json' \) \
  | while read -r f; do d="$OUT/predictions/$(basename "$(dirname "$f")")"; mkdir -p "$d"; cp "$f" "$d/"; done

( cd "$OUT" && zip -qr "$HOME/mura-fm-eval_cluster_files.zip" . )
ls -lh "$HOME/mura-fm-eval_cluster_files.zip"
