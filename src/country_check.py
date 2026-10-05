"""Per-country sanity of submission files: matches per S1, empty rate, candidates per S1.
usage: python country_check.py <dir1> [<dir2> ...]   (each dir has matching_results.tsv + candidate_pairs.tsv)"""
import sys
import polars as pl

from config import WORK_DIR
s1 = pl.read_parquet(WORK_DIR / "raw" / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
for d in sys.argv[1:]:
    rows = []
    for f, col in (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")):
        t = pl.read_csv(f"{d}/{f}", separator="\t", schema_overrides={col: pl.Utf8}).with_columns(pl.col(col).fill_null(""))
        t = t.with_columns(pl.when(pl.col(col) == "").then(0).otherwise(pl.col(col).str.count_matches(",") + 1).alias("n"))
        rows.append(t.join(s1, on="source1_entity_id").group_by("country").agg(
            pl.len().alias("S1"), (pl.col("n").sum() / pl.len()).round(3).alias(f"{f[:4]}_per_S1"),
            ((pl.col("n") == 0).mean() * 100).round(2).alias(f"{f[:4]}_empty_%")))
    out = rows[0].join(rows[1].drop("S1"), on="country").sort("country")
    print(f"== {d}\n{out}")
