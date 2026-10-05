"""TF-IDF/key candidates + dense (embedding) leg. Features are computed only for the NEW pairs; base feature shards are
reused and get the dense similarity / rank joined in.

  python v5_merge.py new                  # CAND_V5/<name>_new.parquet  (dense pairs not in the base candidates)
  (then: BER_CAND_DIR=CAND_V5 BER_FEAT_DIR=FEAT_V5 python pipeline.py features_named <name>_new <split>)
  python v5_merge.py combine <name>       # FEAT_V5/<name>_shards = base shards + dense cols, + new-pair shards
"""
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import polars as pl

from config import WORK_DIR

A = Path(os.environ.get("BER_ROOT", WORK_DIR))
CAND_V4 = A / os.environ.get("M_BASE_CAND", "cand_v4")      # base (TF-IDF/key) candidates + features
FEAT_V4 = A / os.environ.get("M_BASE_FEAT", "feat_v4")
CAND_V5 = A / os.environ.get("M_OUT_CAND", "cand_v5")       # output: base + dense leg
FEAT_V5 = A / os.environ.get("M_OUT_FEAT", "feat_v5")
DENSE = A / os.environ.get("DENSE_TAG", "dense")
NAMES = {"train": "train", "val": "train", "test": "test"}        # name -> split
DCOLS = {"sim_dense": pl.Float32, "rk_dense": pl.Int16}


def new(names=None):
    CAND_V5.mkdir(exist_ok=True)
    for name, split in NAMES.items():
        if names and name not in names:
            continue
        v4 = pl.scan_parquet(CAND_V4 / f"{name}.parquet").select("s1", "cand").collect()
        d = pl.read_parquet(DENSE / f"{split}_pairs.parquet").filter(pl.col("s1").is_in(v4["s1"].unique().implode()))
        n = d.join(v4, on=["s1", "cand"], how="anti")
        n.write_parquet(CAND_V5 / f"{name}_new.parquet")
        print(f"{name}: v4 {len(v4):,} pairs, dense {len(d):,}, new {len(n):,} (+{len(n) / v4['s1'].n_unique():.1f}/S1)",
              flush=True)


_D = None


def _init(path):
    global _D
    _D = pl.read_parquet(path)


def _one(args):
    src, dst, schema = args
    sh = pl.read_parquet(src)
    d = _D.filter(pl.col("s1").is_in(sh["s1"].unique().implode()))       # small right side per shard
    sh = sh.drop([c for c in DCOLS if c in sh.columns]).join(d, on=["s1", "cand"], how="left")
    sh = sh.with_columns([pl.lit(None, t).alias(c) for c, t in schema.items() if c not in sh.columns])
    sh.select([pl.col(c).cast(t) for c, t in schema.items()]).write_parquet(dst)
    return 1


def combine(name: str):
    split = NAMES[name]
    out = FEAT_V5 / f"{name}_shards"
    out.mkdir(parents=True, exist_ok=True)
    old = sorted((FEAT_V4 / f"{name}_shards").glob("*.parquet"))
    newp = sorted((FEAT_V5 / f"{name}_new_shards").glob("*.parquet"))
    schema = dict(pl.read_parquet_schema(old[0]))
    schema.update(DCOLS)
    jobs = [(f, out / f.name, schema) for f in old] + [(f, out / f"n{f.name}", schema) for f in newp]
    # spawn, not fork: a forked child can inherit a locked Polars thread pool and hang forever
    os.environ["POLARS_MAX_THREADS"] = "4"                          # per spawned worker
    ctx = get_context("spawn")
    with ctx.Pool(int(os.environ.get("BER_N_PROCS", 16)), initializer=_init,
                  initargs=(str(DENSE / f"{split}_pairs.parquet"),)) as pool:
        n = sum(pool.imap_unordered(_one, jobs, chunksize=4))
    pl.scan_parquet(out / "*.parquet").sink_parquet(FEAT_V5 / f"{name}.parquet")
    print(f"combined {name}: {n} shards ({len(old)} v4 + {len(newp)} new)", flush=True)


def union(name: str):
    """train/val: base candidates UNION dense pairs (with dense sim/rank) -> OUT_CAND/<name>.parquet;
    features are then computed for the whole union (no base features exist for new train states)."""
    split = NAMES[name]
    CAND_V5.mkdir(exist_ok=True)
    base = pl.read_parquet(CAND_V4 / f"{name}.parquet")
    d = pl.read_parquet(DENSE / f"{split}_pairs.parquet").filter(pl.col("s1").is_in(base["s1"].unique().implode()))
    u = base.join(d, on=["s1", "cand"], how="full", coalesce=True)
    u.write_parquet(CAND_V5 / f"{name}.parquet")
    print(f"union {name}: base {len(base):,} + dense {len(d):,} -> {len(u):,} ({len(u) / base['s1'].n_unique():.1f}/S1)",
          flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "new":
        new(sys.argv[2:])
    elif sys.argv[1] == "union":
        union(sys.argv[2])
    else:
        combine(sys.argv[2])
