#!/bin/bash
#SBATCH -p gpu
# (node pin removed)
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o %x_%j.log
# CE training; settings via --export (CE_MODEL, CE_TAG, CE_TRAIN_FILE, CE_VAL_FILE, CE_MAX_TRAIN, CE_LR, CE_BS)
set -u
source ${AMZML_HOME:-$HOME/amzml}/v6r/env.sh
export BER_WORK_DIR=$W
cd $A/code/src
echo "start $(date) $CE_MODEL -> $CE_TAG"; gpucheck || exit 75
$PY -u cross_encoder.py train && echo "CE_TRAIN_DONE $(date)"
