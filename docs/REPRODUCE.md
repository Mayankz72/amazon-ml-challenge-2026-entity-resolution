# Reproducing the Submission

## 1. Environment

```bash
pip install -r requirements.txt     # Python 3.11
```

Hardware used: a SLURM cluster with V100 32 GB GPUs (dense encoder, cross-encoders) and a large CPU node (features,
LightGBM).

## 2. Layout

The job scripts expect this layout under `AMZML_HOME` (default `~/amzml`):

| Path | Contents |
|---|---|
| `$AMZML_HOME/dataset/{train,test}` | competition data |
| `$AMZML_HOME/code/src` | this repository's `src/` |
| `$AMZML_HOME/v6r` | this repository's `src/scripts/` |
| `$AMZML_HOME/env` | Python environment (`env/bin/python`) |
| `$AMZML_HOME/work` | working directory (`W`) |
| `$TMPDIR/amzml` | large intermediate files (`T`) |

Shared settings (paths, training states) are in `src/scripts/env.sh`.

## 3. India and US (steps 1–5)

From `$AMZML_HOME/code/src`:

```bash
python translit.py               # Indic -> Latin dictionary from the training ground truth
python prep.py train test        # normalised records -> $W/norm
bash ../../v6r/launch.sh         # submits all SLURM jobs below with dependencies
```

| Job | Purpose |
|---|---|
| `cand.sh test4 <country>` / `cand.sh train6 <country>` | TF-IDF and exact-key blocking (test / training + validation states) |
| `dense6.sh` | fine-tune multilingual-e5-small, embed, kNN search (GPU) |
| `feat4.sh` | features of the test TF-IDF candidates |
| `v6.sh` | candidate union, features, stage 1, stage-2 pair lists |
| `cerest.sh` | cross-encoder training pairs |
| `cetrain.sh` ×3 | fine-tune xlm-roberta-large, mdeberta-v3-base, bge-reranker-v2-m3 (GPU) |
| `ce_score_s2.sh` | score the stage-2 pairs with the three cross-encoders (GPU) |
| `v7lite.sh` | stage-2 training and prediction → `$T/out_v7lite/` |

## 4. France (step 6)

From `$AMZML_HOME/code/src`, after `v7lite.sh` has finished (`W=$AMZML_HOME/work`; `<ce>` is `rest_mdeb`,
`rest_xlmrl` or `rest_bge`; for `rest_xlmrl` and `rest_bge` set `MAXTR=500000 BS=32`):

```bash
BER_WORK_DIR=$W python france_aug/fr_test_pairs.py      # France test pairs to rescore
python france_aug/fr_aug.py                              # synthetic French pairs, pass 1
python france_aug/fr_aug2.py                             # synthetic French pairs, pass 2

# per cross-encoder: pass 1, then pass 2 + scoring of the France test pairs
bash france_aug/gpurun.sh <gpu> "<gpu>" $W/ce/<ce> <ce>_fa fraug_train.parquet fraug_val.parquet ""
bash france_aug/gpurun.sh <gpu> "<gpus>" $W/ce/<ce>_fa <ce>_fa2 fraug2_train.parquet fraug2_val.parquet "s2v6_test_fr:3"

SUF=fa2 bash france_aug/fa_predict.sh                    # stage 2 with the French-aware scores
BER_SRC=. python france_aug/mkdmfa2.py                   # France matches -> $AMZML_HOME/fr_fa2_dm.tsv
```

## 5. Final output

```bash
mkdir -p fr
cp $AMZML_HOME/fr_fa2_dm.tsv fr/matching_results.tsv
cp $T/out_v7lite/candidate_pairs.tsv fr/
BER_WORK_DIR=$W python hybrid.py $T/out_v7lite fr France ../output   # India/US rows + France rows
python country_check.py ../output                                    # per-country summary
```

The result is `$AMZML_HOME/code/output/matching_results.tsv` and `candidate_pairs.tsv`.
