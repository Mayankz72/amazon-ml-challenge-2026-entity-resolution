# 3. Results

## Leaderboard

| Metric | Value |
|---|---|
| Public leaderboard F0.5 | **0.98964** |
| Rank | **63** |

## Pipeline

| Stage | Metric | Value |
|---|---|---|
| Blocking | Validation recall: TF-IDF + keys | 98.30% |
| Blocking | Validation recall: + dense leg | 99.63% |
| Stage 1 | Test candidate pairs | 8,770,697 (5.06 per S1) |
| Stage 1 | Validation F0.5 | 0.9817 |
| Stage 2 | Validation F0.5 | **0.9902** |
| Output | Test matches | 5,843,407 (3.37 per S1) |
| Output | S1 with no match | 100,016 |
| Output | France matches | 855,557 |

Validation set: Illinois + Karnataka (145,389 S1), which are not used for training.

## French-aware cross-encoders

AUC on held-out synthetic French pairs (pass 1 validation set):

| Model | Before | After pass 1 |
|---|---|---|
| mdeberta-v3-base | 0.99057 | **0.99886** |
| xlm-roberta-large | 0.99116 | **0.99867** |
| bge-reranker-v2-m3 | 0.99107 | **0.99864** |

## Error analysis

- **False positives:** different businesses at the same address whose names differ by one word, neighbouring
  house or unit numbers, and name-only records with near-identical names.
- **False negatives:** name-only candidates, and heavily rewritten (trade) names at the same address.
