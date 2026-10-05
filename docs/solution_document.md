# ML Challenge 2026: Business Entity Resolution — Solution Document

**Team Name:** mp_up_bangal  
**Team Members:** Mayank Mishra, Akarsh Dubey  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

The pipeline has four parts:

1. **Blocking:** sparse (character TF-IDF and exact keys) and dense (fine-tuned multilingual-e5-small) candidate
   generation.
2. **Stage 1:** a LightGBM candidate filter.
3. **Stage 2:** three multilingual cross-encoders and a context-aware LightGBM.
4. **Decision:** one-to-one assignment, then a per-S1 expected-F0.5 selection.

France does not appear in the training data. For France, the cross-encoders are further fine-tuned on synthetic
French pairs built from labelled training pairs.

**Public leaderboard: 0.98964 (rank 63).**

---

## 2. Methodology

### 2.1 Problem Analysis

- **Data:** the training data covers US and India only; the test set adds France (15% of S1). Each S1 has 0–11
  matches in S2 and S3 (mean 3.46), and 5.6% of S1 have no match. 26% of S2/S3 records match no S1, and no S2/S3
  record matches more than one S1.
- **Name noise:** typos and look-alike characters, word reordering, legal-suffix changes (Pvt/Private, LLC), junk
  prefixes, domain-style names, and Indic-script names.
- **Address noise:** reordered components, abbreviations (Rd/Road), state name vs code, missing or noisy house
  numbers, `Door No` / `Plot No` prefixes, and empty addresses.
- **Hardest cases:** near-identical distractors at the same address (one word or the house number changed), and
  candidates with a name but no address.
- **Validation:** two whole dense states (Illinois + Karnataka, 145,389 S1) that are never used for training.

### 2.2 Solution Strategy

Blocking → stage-1 LightGBM → cross-encoders → stage-2 LightGBM with context features → one-to-one assignment →
per-S1 expected-F0.5 selection.

**Approach Type:** blocking + classifier (sparse and dense blocking; gradient boosting with transformer
cross-encoders).

**Core Innovation:**

1. **Leak-free cross-encoders:** they are trained only on training states that the LightGBM models never see, so
   their scores are out-of-sample features for stage 2.
2. **Competition and context features:** for example, how strongly other S1 records claim the same candidate, and
   the candidate's rank and gap within its S1.
3. **French-aware cross-encoders:** for the country missing from training, they are trained on synthetic French
   pairs that keep their true labels.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used:**

- **TF-IDF:** character 3-gram TF-IDF top-k over name, address, and name + address, run within each country/state.
  It runs forward (S1 → S2/S3, k = 10/15/20), in reverse (S2/S3 → S1, k = 3/3/5), and for S2/S3 records with no
  state (orphans).
- **Exact keys:** identical normalised address; same sorted name + city.
- **Dense leg:** multilingual-e5-small fine-tuned on training matches, with kNN search on `name | address`
  embeddings per country (top 20 S2/S3 per S1, and top 3 S1 per S2/S3 record).

**Candidate pairs generated:** 113.4M TF-IDF/key pairs on test, combined with the dense-leg pairs. The stage-1
LightGBM keeps pairs with p ≥ 0.01, which gives **8,770,697 pairs (5.06 per S1)**. These are exactly the pairs in
`candidate_pairs.tsv`, and exactly the pairs the final model scores.

**How you ensured true matches were not lost:** the legs complement each other (both directions, exact keys, dense
kNN). Validation blocking recall is 98.30% with TF-IDF + keys and **99.63%** with the dense leg added.
Document-frequency pruning is kept loose (`max_df 0.05`) so that common street and city trigrams are retained.

---

## 4. Matching Model

**Features used:**

- **Name:** RapidFuzz ratio, partial ratio, token-sort and token-set ratio, WRatio, Jaro-Winkler and Levenshtein on
  the concatenated name, token Jaccard and containment, length and token-count differences, legal-form agreement,
  and a domain-name flag.
- **Address:** ratio and token-based similarities, numeric-token Jaccard and containment, house-number relation, and
  street, city, state and postcode agreement.
- **Blocking evidence:** the similarity and rank from each blocking leg, exact-key hits, and the dense similarity and
  rank.
- **Stage-2 context:**
  - the pair's rank and gap within its S1;
  - competition from other S1 records for the same candidate;
  - sibling agreement and candidate-cluster statistics;
  - word and number edit log-likelihood ratios.
- **Cross-encoders:** xlm-roberta-large, mdeberta-v3-base and bge-reranker-v2-m3. Each is trained on 1.2M–2M pairs
  sampled from 9.1M pairs (ground-truth positives plus dense-kNN hard negatives) from training states that the
  LightGBM models do not use. Stage 2 uses their scores, their mean, and the mean's rank and gap within the S1 and
  within the candidate.
- **France:** the same cross-encoders are further fine-tuned on synthetic French pairs. These are labelled US pairs
  rewritten identically on both records into French forms: French cities and regions from the test file's France
  addresses, Rue/Av./Bd, LLC → SARL, Inc → SAS, and France-style record noise. Original pairs are mixed in. France
  pairs are scored with these models.

**Model type:**

- **Stage 1:** LightGBM (127 leaves, learning rate 0.05).
- **Stage 2:** LightGBM (63 leaves, learning rate 0.03), using out-of-fold stage-1 scores.
- Both are trained on 9 whole dense states (933,773 S1). The cross-encoder scores are stage-2 features. One
  stage-2 model is used for all countries.

**Threshold selection method:**

1. Each S2/S3 record is kept only for its best S1 (one-to-one).
2. **India and US:** a per-S1 expected-F0.5 selection, tuned on the validation states.
3. **France (no labels):** the top 855,557 pairs by stage-2 probability (≈ 3.3 matches per France S1).

---

## 5. Results & Error Analysis

- **F0.5 (macro):** 0.9902 on validation (Illinois + Karnataka; stage 1 alone scores 0.9817). Public leaderboard:
  **0.98964 (rank 63)**.
- **Common false positives:**
  - near-identical businesses at the same address, with one word changed or added;
  - neighbouring house or unit numbers;
  - records with near-identical names and no address.
- **Common false negatives:**
  - candidates with a name but no address;
  - heavily rewritten (trade) names at an identical address.

---

## 6. Conclusion

Recall comes from complementary sparse and dense blocking. Precision comes from a learned candidate filter,
leak-free cross-encoders, and a stage-2 model of the competition between records. For the unseen country,
synthetic French pairs with true labels let the cross-encoders adapt without any French ground truth.

---

## Appendix

### A. Code Artefacts

| Purpose | Files |
|---|---|
| Normalisation | `src/prep.py`, `src/text.py`, `src/translit.py` |
| Blocking | `src/blocking.py`, `src/dense_block.py`, `src/v5_merge.py` |
| Features and stage 1 | `src/features.py`, `src/pipeline.py` |
| Cross-encoders | `src/cross_encoder.py`, `src/ce_rest.py` |
| Stage 2 and decision | `src/context.py`, `src/edit_features.py`, `src/stage2.py`, `src/decide.py` |
| France step | `src/france_aug/`, merged into the output with `src/hybrid.py` |
| Job scripts | `src/scripts/` (entry point `launch.sh`) |
| Model metadata | `src/models/` |

The run order is in `docs/REPRODUCE.md`.

### B. Additional Results

| Metric | Value |
|---|---|
| Validation blocking recall (TF-IDF + keys → + dense leg) | 98.30% → 99.63% |
| Candidate pairs (test) | 8,770,697 (5.06 per S1) |
| Validation F0.5: stage 1 / stage 2 | 0.9817 / 0.9902 |
| Matches (test) | 5,843,407 (3.37 per S1) |
| S1 with no match (test) | 100,016 |
| France matched pairs | 855,557 |
| Public leaderboard F0.5 | **0.98964** |
| Public leaderboard rank | **63** |

**Models:** LightGBM (MIT), multilingual-e5-small (MIT), mdeberta-v3-base (MIT), xlm-roberta-large (MIT) and
bge-reranker-v2-m3 (Apache-2.0), all under 8B parameters. No external data or lookups were used.
