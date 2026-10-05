#!/bin/bash
# stage 2: stage-1 output + the 3 leak-free cross-encoder scores + CE context features. Needs ce_score_s2.sh.
set -u
source ${AMZML_HOME:-$HOME/amzml}/v6r/env.sh
W6=$T/work_v6; WL=$T/work_v7lite
echo "start $(date)"
until [ -f $A/v6r/CE_REST_SCORED ]; do sleep 60; done
echo "inputs ready $(date)"
mkdir -p $WL $T/out_v7lite $A/v7lite
for x in norm raw indic_dict.json ce; do [ -e $WL/$x ] || ln -s $W/$x $WL/$x; done
for f in lgb_v1.json lgb_v1.txt s2_oof_train.npy test_scored.parquet; do cp $W6/$f $WL/; done
xl=$(ls $W/ce/rest_xlmrl/s2v6_*_scores_*.parquet | paste -sd,)
md=$(ls $W/ce/rest_mdeb/s2v6_*_scores_*.parquet | paste -sd,)
bg=$(ls $W/ce/rest_bge/s2v6_*_scores_*.parquet | paste -sd,)
export BER_ROOT=$W BER_WORK_DIR=$WL BER_OUT_DIR=$T/out_v7lite BER_TRAIN_STATES=$V6_STATES BER_CAND_DIR=$T/cand_v6u \
       BER_FEAT_DIR=$T/feat_v6u BER_N_JOBS=${NCPU:-40} BER_N_PROCS=16 POLARS_MAX_THREADS=${NCPU:-40} \
       S2_CE_EXTRA="xl=$xl;md=$md;bg=$bg" S2_CE_CTX=1
cd $A/code/src
$PY -u stage2.py fit || exit 1
$PY -u stage2.py predict || exit 1
cp $T/out_v7lite/matching_results.tsv $T/out_v7lite/candidate_pairs.tsv $WL/lgb_s2.json $A/v7lite/
echo "V7LITE_DONE $(date)"
