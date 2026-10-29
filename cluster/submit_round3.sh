#!/bin/bash
cd ~/mediinsight_project/codes || exit 1
for f in run_round3_320_s*_*.pbs; do echo "$f -> $(qsub $f)"; done
qstat -u $USER
