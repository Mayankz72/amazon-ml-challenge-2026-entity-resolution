# 1. Problem Statement

## Task

The data contains business records from three independent sources that share no identifiers.

- **Source 1 (S1)** is the deduplicated reference source.
- For each S1 record, the task is to return every record in **Source 2 (S2)** and **Source 3 (S3)** that describes
  the same real-world business.
- An S1 record can have zero, one or several matches.

## Data

Each source file has four columns:

| Column | Description |
|---|---|
| `entity_id` | unique ID; the prefix `S1-`, `S2-` or `S3-` gives the source |
| `business_name` | may contain abbreviations, legal suffixes, typos and transliterations |
| `business_address` | may be partial, reordered, or missing components |
| `country` | `US` and `India` in train; the test set also contains `France` |

The training ground truth maps each `source1_entity_id` to a comma-separated list of matched S2/S3 IDs.

| Split | S1 | S2 | S3 |
|---|---|---|---|
| Train | 2.21M | 5.03M | 5.29M |
| Test | 1.73M | 4.89M | 5.08M |

## Key observations

| Observation | Value | Implication |
|---|---|---|
| Test countries | India 47%, US 38%, France 15% | the model must generalise to an unseen country |
| S1 records with no match | 5.6% | an explicit "no match" decision is needed |
| Matches per S1 | mean 3.46, max 11 | recall within each S1 matters |
| S2/S3 records matched to more than one S1 | 0 | a one-to-one constraint can be applied |
| S2/S3 records matching no S1 | 26% | many hard negatives |
| Cross-country matches | 0% | blocking within each country |
| True pairs with identical name / address | 10.7% / 7.2% | fuzzy matching is required |
| Non-Latin names in India | ~9% | transliteration is required |

**Typical noise:** character typos (`5ervices`, `lce`), shuffled words, junk prefixes (`<<`, `##`), added, dropped
or reordered legal suffixes (LLC, Pvt Ltd), domain-style names (`example.com`), Indic-script names, reordered
address components, abbreviations (Rd/Road), state names vs codes, noisy house numbers and empty addresses.

## Output

Two tab-separated files:

1. `matching_results.tsv`: one row per test S1 (`source1_entity_id`, `matched_entity_ids`). This file is scored.
2. `candidate_pairs.tsv`: the candidate set scored by the final model. Every match must also be a candidate.

## Metric

The metric is macro-averaged **F0.5** per S1 record, which weights precision twice as much as recall:

```
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

An S1 record with no true match scores 1.0 if the prediction is empty and 0.0 otherwise.

## Rules

- No external data, APIs or lookups.
- Models must be MIT or Apache-2.0 licensed, with at most 8B parameters.
