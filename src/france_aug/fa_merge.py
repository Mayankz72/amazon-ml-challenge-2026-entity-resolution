# CE score files for predict: France test pairs from the French-augmented CE (<tag>_fa), everything else unchanged

import os as _os
AMZ = _os.environ.get("AMZML_HOME", _os.path.expanduser("~/amzml"))
TMP = _os.environ.get("AMZML_TMP", "/tmp/amzml")
import glob, os, polars as pl
SUF = os.environ.get("SUF", "fa")
W = AMZ + "/work/ce/"
for t in ["rest_xlmrl", "rest_mdeb", "rest_bge"]:
    old = pl.concat([pl.read_parquet(f) for f in glob.glob(W + f"{t}/s2v6_test_scores_*.parquet")]).unique(["s1", "cand"])
    new = pl.concat([pl.read_parquet(f) for f in glob.glob(W + f"{t}_{SUF}/s2v6_test_fr_scores_*.parquet")]).unique(["s1", "cand"])
    m = pl.concat([old.join(new.select("s1", "cand"), on=["s1", "cand"], how="anti"), new.select(old.columns)])
    m.write_parquet(W + f"{t}_{SUF}/s2v6_test_merged.parquet")
    j = new.join(old, on=["s1", "cand"], suffix="_old")
    print(f"{t}: test {len(old):,} -> merged {len(m):,}; France pairs replaced {len(new):,}; "
          f"FR mean score old {j['ce_old'].mean():.4f} -> new {j['ce'].mean():.4f}; corr {j.select(pl.corr('ce','ce_old')).item():.4f}", flush=True)
