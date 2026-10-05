#!/bin/bash
#SBATCH -J cerest
#SBATCH -p gpu
# (node pin removed)
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -o cerest_%j.log
# CE training pairs from the training states the LightGBM models use neither for training nor validation (leak-free)
set -u
source ${AMZML_HOME:-$HOME/amzml}/v6r/env.sh
mkdir -p $T/dense_rest $W/ce
[ -d $T/dense_rest/model ] || cp -r $T/dense_v6/model $T/dense_rest/
export BER_WORK_DIR=$W DENSE_TAG=$T/dense_rest DENSE_STATES=$V6_STATES,il,ka DENSE_STATES_INVERT=1 \
       DENSE_K=10 DENSE_K_REV=2 POLARS_MAX_THREADS=16
cd $A/code/src
echo "start $(date)"; gpucheck || exit 75
for s in 1 2 3; do [ -f $T/dense_rest/train_s$s.npy ] || $PY -u dense_block.py embed train $s || exit 1; done
[ -f $T/dense_rest/train_pairs.parquet ] || $PY -u dense_block.py search train || exit 1
[ -f $W/ce/rest_val.parquet ] || $PY -u ce_rest.py || exit 1
echo "CEREST_PAIRS_DONE $(date)"
