# 2. Methodology

```
Normalisation → Blocking (sparse + dense) → Stage-1 LightGBM → Cross-encoders → Stage-2 LightGBM
              → One-to-one assignment → Per-S1 expected-F0.5 selection
```

## 2.1 Data split

Training data is split by whole **states**. Keeping states whole preserves the real density of candidates and
look-alike businesses, and it keeps the competition features in stage 2 complete.

| Use | States |
|---|---|
| Stage 1 and stage 2 (LightGBM) | 9 dense states: VA, TN, KL, MH, DL, TX, NY, NC, UP (933,773 S1) |
| Validation | Illinois + Karnataka (145,389 S1) |
| Dense encoder and cross-encoders | all remaining training states |

## 2.2 Normalisation (`prep.py`, `text.py`, `translit.py`)

Each record is turned into a set of derived fields:

- **Name:** lower-cased, accents removed, transliterated, and look-alike characters fixed (`5ervices` → `services`).
  Legal forms are mapped to a canonical form and stored separately. This gives `name_core` (the name without legal
  and filler words), `name_sorted` and `name_concat`, plus a flag for domain-style names.
- **Address:** street types and ordinals are standardised, and the address is parsed into house number, street,
  city, state code and postcode. All numeric tokens are also kept as a separate field.
- **Transliteration:** an Indic-script → Latin token dictionary is learned from the training ground truth only
  (Indic names are word-by-word renderings of the S1 name).

## 2.3 Candidate generation (`blocking.py`, `dense_block.py`, `v5_merge.py`)

**Sparse legs:** character 3-gram TF-IDF (`char_wb`, `max_df 0.05`) over three views: name, address, and
name + address. The search runs inside each (country, state) partition.

| Leg | Direction | top-k (name / address / name + address) |
|---|---|---|
| Forward | S1 → S2/S3 | 10 / 15 / 20 |
| Reverse | S2/S3 → S1 | 3 / 3 / 5 |
| Orphan | S2/S3 records with no state → S1 of the whole country | 5 (name) |
| Exact key | identical normalised address; same sorted name + city | buckets ≤ 30 |

**Dense leg:** multilingual-e5-small is fine-tuned with a contrastive loss (InfoNCE with in-batch negatives) on
training matches. The input text is `name | address`. Neighbours come from a kNN search per country: the top 20
S2/S3 records for each S1, and the top 3 S1 for each S2/S3 record.

| Blocking (validation) | Recall |
|---|---|
| TF-IDF + keys | 98.30% |
| + dense leg | **99.63%** |

## 2.4 Stage 1 (`features.py`, `pipeline.py`)

LightGBM (127 leaves, learning rate 0.05) on 60 pairwise features:

- **Name:** RapidFuzz ratio, partial ratio, token-sort and token-set ratio, WRatio, Jaro-Winkler and Levenshtein
  similarity on the concatenated name, token Jaccard and containment, length and token-count differences, legal-form
  agreement, a domain-name flag, and the name's similarity to the other record's address.
- **Address:** ratio, token-set and token-sort ratio, partial ratio, Jaccard and containment; numeric-token Jaccard
  and containment; house-number relation (equal, missing, prefix, different); street, city, state and postcode
  agreement.
- **Blocking evidence:** the similarity and rank from every blocking leg, exact-key hits, and the dense similarity
  and rank.

Pairs with p ≥ 0.01 are kept. On test this leaves **8,770,697 pairs (5.06 per S1)**, which are exactly the pairs in
`candidate_pairs.tsv`.

## 2.5 Cross-encoders (`cross_encoder.py`, `ce_rest.py`)

Each cross-encoder reads both records together as `name | address` (S1) and `name | address` (candidate), up to 96
tokens:

| Model | Training sample | Learning rate |
|---|---|---|
| xlm-roberta-large | 1.2M pairs | 1e-5 |
| mdeberta-v3-base | 2.0M pairs | 2e-5 |
| bge-reranker-v2-m3 | 1.2M pairs | 1e-5 |

The training pairs come from training states that the LightGBM models never use. Positives are all ground-truth
matches; hard negatives are dense-kNN neighbours (rank ≤ 8). Because these states are never seen by the LightGBM
models, the cross-encoder scores carry no label leakage when used as stage-2 features.

## 2.6 Stage 2 (`context.py`, `edit_features.py`, `stage2.py`)

LightGBM (63 leaves, learning rate 0.03) on the stage-1 features. To avoid leakage, the stage-1 probability on the
training data is out-of-fold (2 folds of whole states). The model adds these context features:

- **Within S1:** the pair's rank and gap to the best score, the number of strong candidates, and the score sum.
- **Competition:** how many S1 records claim the same candidate, the best rival probability, and the margin.
- **Siblings:** name, address and house-number agreement with the S1's other likely matches.
- **Candidate cluster:** size, mean and maximum probability of the linked group, and the strongest rival.
- **Edit evidence:** which words or numbers differ between the two records, scored with log-likelihood ratios
  learned from training pairs. These separate noise (typos, generic words) from real differences (a changed word or
  house number).
- **Cross-encoders:** the three scores, their mean, and the mean's rank and gap within the S1 and within the
  candidate.

## 2.7 Decision (`decide.py`, `context.py`)

1. **One-to-one:** each S2/S3 record is kept only for the S1 that gives it the highest score.
2. **Expected F0.5:** for each S1, candidates are sorted by probability and the top-k with the highest expected
   F0.5 is chosen. If the probability of having no match is higher, the prediction is empty.

## 2.8 France (`france_aug/`)

France makes up 15% of the test S1 but does not appear in the training data. We therefore build **synthetic French
pairs with true labels**:

1. **Pass 1 (`fr_aug.py`):** labelled US pairs are rewritten into French forms. The rewrite is identical on both
   records, so every label stays correct. It uses French cities and regions taken from the test file's France
   addresses, French street types (Rue, Av., Bd …), legal forms (LLC → SARL, Inc → SAS) and business words
   (Sons → Fils, School → École …). The training set is 500k French + 300k original pairs.
2. **Pass 2 (`fr_aug2.py`):** France-style record noise is added to the S2/S3 side: house-number formats (N°, bis,
   ter), region ↔ département, dropped accents, upper case, and street abbreviations. The training set is 700k
   French + 300k original pairs.
3. Each cross-encoder continues training on pass 1, then on pass 2.
4. The France test pairs are rescored with these cross-encoders and the stage-2 model predicts. France keeps the
   top 855,557 one-to-one pairs (≈ 3.3 per S1). These rows replace the France rows of the output (`hybrid.py`);
   India and US rows are unchanged.

## 2.9 Compute

A SLURM node with V100 32 GB GPUs ran the dense encoder and the cross-encoders. Features and LightGBM ran on CPU
(Polars, RapidFuzz, sharded processes). The seed is fixed (`SEED = 42`).
