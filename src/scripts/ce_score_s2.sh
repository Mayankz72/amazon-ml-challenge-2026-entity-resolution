#!/bin/bash
# score the stage-2 pairs (train/val/test) with the 3 leak-free rest CEs (one GPU; can be sharded with CE_SHARD)
set -u
source ${AMZML_HOME:-$HOME/amzml}/v6r/env.sh
until [ -f $A/v6r/S2PAIRS_READY ]; do sleep 60; done
cd $A/code/src
for t in rest_xlmrl rest_mdeb rest_bge; do
  for s in s2v6_train s2v6_val s2v6_test; do
    BER_WORK_DIR=$W CE_TAG=$t CE_SHARD=0/1 $PY -u cross_encoder.py score $s || exit 1
  done
done
touch $A/v6r/CE_REST_SCORED
echo "CE rest scored $(date)"
