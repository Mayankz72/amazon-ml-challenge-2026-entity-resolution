#!/bin/bash
#SBATCH -J stage1
#SBATCH -p gpu
# (node pin removed)
#SBATCH -c 64
#SBATCH --mem=700G
#SBATCH -o v6_%j.log
# candidate union (TF-IDF/key + dense leg) + pair features + stage-1 LightGBM (9 dense training states, 933k S1)
# -> candidate_pairs set + stage-2 pair lists for cross-encoder scoring
# submitted with --dependency=afterok on the candidate, dense and feat4 jobs
set -u
source ${AMZML_HOME:-$HOME/amzml}/v6r/env.sh
W6=$T/work_v6
mkdir -p $W6 $T/out_v6 $A/v6 $W/ce
for x in norm raw indic_dict.json ce; do [ -e $W6/$x ] || ln -s $W/$x $W6/$x; done
export BER_ROOT=$W BER_WORK_DIR=$W6 BER_OUT_DIR=$T/out_v6 BER_TRAIN_STATES=$V6_STATES \
       BER_N_JOBS=${NCPU:-64} BER_N_PROCS=24 POLARS_MAX_THREADS=${NCPU:-64} DENSE_TAG=$T/dense_v6
cd $A/code/src
feats() {   # $1 name, $2 split, $3 shard processes  (BER_CAND_DIR / BER_FEAT_DIR from caller)
  for ((k = 0; k < $3; k++)); do
    ( export BER_FEAT_SHARD=$k/$3 BER_N_JOBS=3 POLARS_MAX_THREADS=3
      while true; do $PY -u pipeline.py features_named $1 $2; rc=$?; [ $rc -eq 3 ] || break; done ) &
  done
  wait
  while true; do $PY -u pipeline.py features_named $1 $2; rc=$?; [ $rc -eq 0 ] && return 0; [ $rc -eq 3 ] || return $rc; done
}
echo "start $(date) on $(hostname)"; df -h /tmp | tail -1
# 1. candidates: merge train/val country shards, union with dense; test = TF-IDF/key test + new dense pairs
[ -f $T/cand_v6/val.parquet ] || BER_CAND_DIR=$T/cand_v6 $PY -u pipeline.py candidates train || exit 1
for n in train val; do [ -f $T/cand_v6u/$n.parquet ] || M_BASE_CAND=$T/cand_v6 M_OUT_CAND=$T/cand_v6u $PY -u v5_merge.py union $n || exit 1; done
[ -f $T/cand_v6u/test_new.parquet ] || M_BASE_CAND=$T/cand_v4 M_OUT_CAND=$T/cand_v6u $PY -u v5_merge.py new test || exit 1
# 2. features
export BER_CAND_DIR=$T/cand_v6u BER_FEAT_DIR=$T/feat_v6u
[ -f $T/feat_v6u/train.parquet ] || feats train train 24 || exit 1
[ -f $T/feat_v6u/val.parquet ] || feats val train 24 || exit 1
[ -f $T/feat_v6u/test_new.parquet ] || feats test_new test 24 || exit 1
[ -f $T/feat_v6u/test.parquet ] || M_BASE_FEAT=$T/feat_v4 M_OUT_FEAT=$T/feat_v6u $PY -u v5_merge.py combine test || exit 1
echo "features ready $(date)"
# 3. stage 1
$PY -u pipeline.py fit || exit 1
$PY -u pipeline.py predict || exit 1
$PY -u stage2.py pairs || exit 1
echo "STAGE1_DONE $(date)"
# 4. export the stage-2 pairs for cross-encoder scoring (ce_score_s2.sh)
for n in train val test; do
  [ -f $W/ce/s2v6_$n.parquet ] || $PY -u cross_encoder.py export_pairs $W6/s2_pairs_$n.parquet $([ $n = test ] && echo test || echo train) s2v6_$n || exit 1
done
touch $A/v6r/S2PAIRS_READY
echo "V6_DONE $(date)"
