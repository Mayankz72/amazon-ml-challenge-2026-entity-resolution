"""Cross-encoder training pairs from the training states that the LightGBM models use neither for training nor validation.
Hard negatives = dense kNN neighbours (rank <= K); positives = all ground-truth matches of those S1.
-> CE/rest_train.parquet, CE/rest_val.parquet (s1, cand, y, text_a, text_b)."""
import os
import sys

import numpy as np
import polars as pl

from config import SEED, WORK_DIR
from data import load_ground_truth

DENSE = WORK_DIR / os.environ["DENSE_TAG"]
CE = WORK_DIR / "ce"
K = int(os.environ.get("REST_K", 8))

d = pl.read_parquet(DENSE / "train_pairs.parquet").filter(pl.col("rk_dense") <= K).select("s1", "cand")
s1 = d["s1"].unique()
gt = load_ground_truth().drop_nulls().rename({"source1_entity_id": "s1", "match": "cand"}) \
                        .filter(pl.col("s1").is_in(s1.implode()))
pairs = pl.concat([d, gt]).unique().join(gt.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "cand"], how="left") \
          .with_columns(pl.col("y").fill_null(0))
raw = lambda s: pl.scan_parquet(WORK_DIR / "raw" / f"train_source{s}.parquet")
txt = lambda df, col: df.select(pl.col("entity_id").alias(col), pl.concat_str(
    ["business_name", "business_address"], separator=" | ").alias("text_" + ("a" if col == "s1" else "b")))
r1 = raw(1).filter(pl.col("entity_id").is_in(s1.implode())).collect()
rp = pl.concat([raw(s).filter(pl.col("entity_id").is_in(pairs["cand"].unique().implode())).collect() for s in (2, 3)])
pairs = pairs.join(txt(r1, "s1"), on="s1", how="left").join(txt(rp, "cand"), on="cand", how="left")
rng = np.random.default_rng(SEED)
val_s1 = set(rng.choice(s1.to_numpy(), size=min(20_000, len(s1)), replace=False).tolist())
is_val = pairs["s1"].is_in(list(val_s1))
pairs.filter(~is_val).write_parquet(CE / "rest_train.parquet")
pairs.filter(is_val).write_parquet(CE / "rest_val.parquet")
print(f"rest pairs: {len(pairs):,} (pos {int(pairs['y'].sum()):,}) from {len(s1):,} S1; val S1 {len(val_s1):,}", flush=True)
