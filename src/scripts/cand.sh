#!/bin/bash
#SBATCH -p gpu
# (node pin removed)
#SBATCH -o %x_%j.log
# candidates for one country:  cand.sh test4 <Country>  (test blocking)  |  cand.sh train6 <Country>  (train/val states)
set -u
source ${AMZML_HOME:-$HOME/amzml}/v6r/env.sh
mkdir -p $T/work_$1_$2
for x in norm raw indic_dict.json; do [ -e $T/work_$1_$2/$x ] || ln -s $W/$x $T/work_$1_$2/$x; done
export BER_WORK_DIR=$T/work_$1_$2 BER_N_JOBS=${NCPU:-32} BER_N_PROCS=12 POLARS_MAX_THREADS=${NCPU:-32} \
       BER_COUNTRIES=$2
cd $A/code/src
echo "start $1 $2 $(date) on $(hostname)"; df -h /tmp | tail -1
case $1 in
  test4)  BER_CAND_DIR=$T/cand_v4 $PY -u pipeline.py candidates test || exit 1 ;;
  train6) BER_CAND_DIR=$T/cand_v6 BER_TRAIN_STATES=$V6_STATES $PY -u pipeline.py candidates train || exit 1 ;;
esac
echo "DONE $1 $2 $(date)"
