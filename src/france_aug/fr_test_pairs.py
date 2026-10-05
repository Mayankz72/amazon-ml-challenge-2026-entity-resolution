"""France subset of the stage-2 test pairs (ce/s2v6_test_fr.parquet), rescored by the French-aware cross-encoders.
usage (from src/): BER_WORK_DIR=<work> python france_aug/fr_test_pairs.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polars as pl
from config import WORK_DIR
from prep import scan_norm

CE = WORK_DIR / "ce"
ids = scan_norm("test", 1, ["entity_id", "country"]).filter(pl.col("country") == "France")["entity_id"]
te = pl.read_parquet(CE / "s2v6_test.parquet").filter(pl.col("s1").is_in(ids.implode()))
te.write_parquet(CE / "s2v6_test_fr.parquet")
print(f"France stage-2 test pairs to rescore: {len(te):,}", flush=True)
