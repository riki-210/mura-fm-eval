#!/bin/bash
cd ~/mediinsight_project/codes || exit 1
for f in run_round4_224_s*_*.pbs; do echo "$f -> $(qsub $f)"; done
qstat -u $USER
