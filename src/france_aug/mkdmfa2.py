"""France rows from the French-augmented-CE stage 2 with the threshold chosen so France keeps a fixed
pair budget of 855,557 (~3.3 per France S1). One-to-one first (same as all decisions)."""
import os as _os
AMZ = _os.environ.get("AMZML_HOME", _os.path.expanduser("~/amzml"))
TMP = _os.environ.get("AMZML_TMP", "/tmp/amzml")

import polars as pl, sys, os
sys.path.insert(0, os.environ["BER_SRC"])
from decide import one_to_one, write_lists
from prep import scan_norm
T = TMP + "/"; N = 855_557
fr = scan_norm("test", 1, ["entity_id", "country"]).filter(pl.col("country") == "France")["entity_id"]
d = pl.read_parquet(f"{T}work_fa2/test_scored_s2.parquet").select("s1", "cand", "p2").filter(pl.col("s1").is_in(fr.implode()))
d = one_to_one(d, "p2").sort("p2", descending=True)
sel = d.head(N)
print(f"threshold for {N:,} pairs: p2 >= {sel['p2'].min():.4f}; S1 with a match {sel['s1'].n_unique():,} of {len(fr):,}")
write_lists(fr.to_list(), sel.select("s1", "cand"), AMZ + "/fr_fa2_dm.tsv", "matched_entity_ids")
print("DM_DONE")
