#!/bin/bash
# stage 2 (already trained) + French-augmented CE scores on France test pairs -> predict
set -u
source ${AMZML_HOME:-$HOME/amzml}/v6r/env.sh
SUF=${SUF:-fa}; WL=$T/work_v7lite; WF=$T/work_$SUF
mkdir -p $WF $T/out_$SUF $A/$SUF
for x in norm raw indic_dict.json ce; do [ -e $WF/$x ] || ln -s $W/$x $WF/$x; done
for f in lgb_v1.json lgb_v1.txt s2_oof_train.npy test_scored.parquet lgb_s2.txt lgb_s2.json edit_llr_tables.json fit_s2.done; do cp $WL/$f $WF/; done
$PY -u $A/code/src/france_aug/fa_merge.py || exit 1
f() { echo "$(ls $W/ce/$1/s2v6_train_scores_*.parquet $W/ce/$1/s2v6_val_scores_*.parquet | paste -sd,),$W/ce/$1_$SUF/s2v6_test_merged.parquet"; }
export BER_ROOT=$W BER_WORK_DIR=$WF BER_OUT_DIR=$T/out_$SUF BER_TRAIN_STATES=$V6_STATES BER_CAND_DIR=$T/cand_v6u \
       BER_FEAT_DIR=$T/feat_v6u BER_N_JOBS=24 BER_N_PROCS=16 POLARS_MAX_THREADS=24 \
       S2_CE_EXTRA="xl=$(f rest_xlmrl);md=$(f rest_mdeb);bg=$(f rest_bge)" S2_CE_CTX=1
cd $A/code/src
$PY -u stage2.py predict || exit 1
cp $T/out_$SUF/matching_results.tsv $A/$SUF/
echo "FA_DONE $(date)"
