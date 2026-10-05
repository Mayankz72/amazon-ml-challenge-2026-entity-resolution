#!/bin/bash
#SBATCH -J feat4
#SBATCH -p gpu
# (node pin removed)
#SBATCH -c 48
#SBATCH --mem=400G
#SBATCH -o feat4_%j.log
# merge the test candidate shards (India/US/France) and compute their features
set -u
source ${AMZML_HOME:-$HOME/amzml}/v6r/env.sh
mkdir -p $T/work_feat4
for x in norm raw indic_dict.json; do [ -e $T/work_feat4/$x ] || ln -s $W/$x $T/work_feat4/$x; done
export BER_WORK_DIR=$T/work_feat4 BER_CAND_DIR=$T/cand_v4 BER_FEAT_DIR=$T/feat_v4 BER_N_JOBS=${NCPU:-48} BER_N_PROCS=16 POLARS_MAX_THREADS=${NCPU:-48}
cd $A/code/src
echo "start $(date)"
$PY -u pipeline.py candidates test || exit 1          # merges the country shards
for ((k = 0; k < 20; k++)); do
  ( export BER_FEAT_SHARD=$k/20 BER_N_JOBS=3 POLARS_MAX_THREADS=3
    while true; do $PY -u pipeline.py features test; rc=$?; [ $rc -eq 3 ] || break; done ) &
done
wait
while true; do $PY -u pipeline.py features test; rc=$?; [ $rc -eq 0 ] && break; [ $rc -eq 3 ] || exit $rc; done
ls -la $T/cand_v4 $T/feat_v4
echo "FEAT4_DONE $(date)"
