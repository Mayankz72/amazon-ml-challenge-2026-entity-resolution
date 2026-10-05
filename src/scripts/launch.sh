#!/bin/bash
# steps 1-5 of docs/REPRODUCE.md: submits all jobs with dependencies (sbatch); final India/US output from v7lite.sh
cd ${AMZML_HOME:-$HOME/amzml}/v6r
rm -f S2PAIRS_READY
j() { sbatch --parsable "$@"; }
t1=$(j -J c4India -c 32 --mem=300G cand.sh test4 India)
t2=$(j -J c4US -c 32 --mem=300G cand.sh test4 US)
t3=$(j -J c4France -c 16 --mem=150G cand.sh test4 France)
c1=$(j -J c6India -c 32 --mem=300G cand.sh train6 India)
c2=$(j -J c6US -c 32 --mem=300G cand.sh train6 US)
d=$(j dense6.sh)
f=$(j --dependency=afterok:$t1:$t2:$t3 feat4.sh)
v=$(j --dependency=afterok:$c1:$c2:$d:$f v6.sh)
r=$(j --dependency=afterok:$d cerest.sh)
x=$(j --dependency=afterok:$r -J cexl --export=ALL,CE_MODEL=FacebookAI/xlm-roberta-large,CE_TAG=rest_xlmrl,CE_TRAIN_FILE=rest_train.parquet,CE_VAL_FILE=rest_val.parquet,CE_MAX_TRAIN=1200000,CE_LR=1e-5,CE_BS=32 cetrain.sh)
m=$(j --dependency=afterok:$r -J cemd --export=ALL,CE_MODEL=microsoft/mdeberta-v3-base,CE_TAG=rest_mdeb,CE_TRAIN_FILE=rest_train.parquet,CE_VAL_FILE=rest_val.parquet,CE_MAX_TRAIN=2000000 cetrain.sh)
b=$(j --dependency=afterok:$r -J cebge --export=ALL,CE_MODEL=BAAI/bge-reranker-v2-m3,CE_TAG=rest_bge,CE_TRAIN_FILE=rest_train.parquet,CE_VAL_FILE=rest_val.parquet,CE_MAX_TRAIN=1200000,CE_LR=1e-5,CE_BS=32 cetrain.sh)
c=$(j --dependency=afterok:$x:$m:$b:$v --gres=gpu:1 -c 6 --mem=64G ce_score_s2.sh)
s=$(j --dependency=afterok:$c v7lite.sh)
echo "c4 $t1 $t2 $t3 | c6 $c1 $c2 | dense6 $d | feat4 $f | v6 $v | cerest $r | cexl $x | cemd $m | cebge $b | cescore $c | v7lite $s"
