#!/bin/bash
# run CE train + sharded scoring directly on given free GPUs inside a user allocation (node CPUs fully allocated)
# usage: gpurun.sh <train_gpu> "<score gpus>" <CE_MODEL> <CE_TAG> <TRAIN_FILE> <VAL_FILE> "<NAME:nshards> ..."
set -u
source ${AMZML_HOME:-$HOME/amzml}/v6r/env.sh
export BER_WORK_DIR=$W
TG=$1; SG=($2); MODEL=$3; TAG=$4; TRF=$5; VAF=$6; JOBS=($7)
cd $A/code/src
echo "train $TAG on gpu $TG $(date)"
CUDA_VISIBLE_DEVICES=$TG CE_MODEL=$MODEL CE_TAG=$TAG CE_TRAIN_FILE=$TRF CE_VAL_FILE=$VAF CE_MAX_TRAIN=${MAXTR:-3000000} CE_LR=1e-5 CE_BS=${BS:-64} \
  $PY -u cross_encoder.py train || exit 1
echo "CE_TRAIN_DONE $TAG $(date)"
# build shard list, distribute round-robin over score GPUs
L=(); for j in "${JOBS[@]}"; do n=${j%%:*}; k=${j##*:}; for s in $(seq 0 $((k-1))); do L+=("$n $s $k"); done; done
for g in $(seq 0 $((${#SG[@]}-1))); do
  ( for i in $(seq $g ${#SG[@]} $((${#L[@]}-1))); do set -- ${L[$i]}
      CUDA_VISIBLE_DEVICES=${SG[$g]} CE_TAG=$TAG CE_SHARD=$2/$3 $PY -u cross_encoder.py score $1 > $A/gs_${TAG}_$1_$2.log 2>&1
    done ) &
done
wait
echo "SCORE_DONE $TAG $(date)"
