#!/bin/bash
#SBATCH -J dense6
#SBATCH -p gpu
# (node pin removed)
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -o dense6_%j.log
# dense leg: e5-small fine-tuned WITHOUT any train/val-state S1, embeddings for train+val states and test
set -u
source ${AMZML_HOME:-$HOME/amzml}/v6r/env.sh
mkdir -p $T/dense_v6
export BER_WORK_DIR=$W DENSE_TAG=$T/dense_v6 DENSE_EXCLUDE_STATES=1 DENSE_STATES=$V6_STATES,il,ka POLARS_MAX_THREADS=16
cd $A/code/src
echo "start $(date)"; nvidia-smi -L; gpucheck || exit 75
[ -f $T/dense_v6/model/config.json ] || $PY -u dense_block.py finetune || exit 1
for s in 1 2 3; do [ -f $T/dense_v6/train_s$s.npy ] || $PY -u dense_block.py embed train $s || exit 1; done
$PY -u dense_block.py search train || exit 1
for s in 1 2 3; do [ -f $T/dense_v6/test_s$s.npy ] || $PY -u dense_block.py embed test $s || exit 1; done
$PY -u dense_block.py search test || exit 1
echo "DENSE_DONE $(date)"
