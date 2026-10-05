"""Per-country hybrid of two submissions with the SAME candidate_pairs.tsv:
python hybrid.py <base_dir> <other_dir> <country> <out_dir>  -> rows of <country> S1 taken from other_dir."""
import shutil, sys, filecmp
import polars as pl
base, other, country, out = sys.argv[1:5]
assert filecmp.cmp(f"{base}/candidate_pairs.tsv", f"{other}/candidate_pairs.tsv", shallow=False), "candidate files differ"
from config import WORK_DIR
s1 = pl.read_parquet(WORK_DIR / "raw" / "test_source1.parquet", columns=["entity_id", "country"])
ids = s1.filter(pl.col("country") == country)["entity_id"]
rd = lambda d: pl.read_csv(f"{d}/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
b, o = rd(base), rd(other)
res = b.filter(~pl.col("source1_entity_id").is_in(ids.implode())).vstack(o.filter(pl.col("source1_entity_id").is_in(ids.implode())))
res = b.select("source1_entity_id").join(res, on="source1_entity_id", how="left")        # keep base row order
import os; os.makedirs(out, exist_ok=True)
res.write_csv(f"{out}/matching_results.tsv", separator="\t", null_value="")
shutil.copy(f"{base}/candidate_pairs.tsv", f"{out}/candidate_pairs.tsv")
print(f"hybrid: {len(res):,} rows, {country} rows from {other}: {res.filter(pl.col('source1_entity_id').is_in(ids.implode())).height:,}")
