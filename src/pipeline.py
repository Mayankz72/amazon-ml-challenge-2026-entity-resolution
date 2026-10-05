"""End-to-end pipeline: blocking -> pair features -> LightGBM -> decision -> submission files.

Usage:
  python pipeline.py candidates train   # blocking for a train sample (train + val S1 subsets)
  python pipeline.py candidates test    # blocking for all test S1
  python pipeline.py features train|test
  python pipeline.py fit                # train LightGBM on train pairs, tune threshold on val pairs
  python pipeline.py predict            # score test pairs, write output/*.tsv
"""
import gc
import json
import os
import sys
from pathlib import Path
import time
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import BLOCK_COLS, STATE_GROUP, run_country
from config import N_JOBS, N_PROCS, OUT_DIR, SEED, WORK_DIR
from data import load_ground_truth
from decide import one_to_one, sweep, write_lists
from features import STR_COLS, pair_features
from prep import normalise, scan_norm

N_TRAIN_S1 = int(os.environ.get("BER_N_TRAIN_S1", 150_000))
N_VAL_S1 = 50_000
TRAIN_STATE_CAP = 80_000     # train may use bigger states (cost of blocking grows with state size)
# Whole dense states (dense cities dominate test): val = Illinois + Karnataka, train = Virginia, Tamil Nadu, Kerala.
EXPLICIT_STATES = ({"val": os.environ.get("BER_VAL_STATES", "il,ka").split(","),
                    "train": os.environ.get("BER_TRAIN_STATES", "va,tn,kl").split(",")}   # 9 dense training states via env.sh
                   if os.environ.get("BER_DENSE_SPLIT", "1") == "1" else None)
CAND = Path(os.environ.get("BER_CAND_DIR", WORK_DIR / "cand"))     # override to run experiments side by side
FEAT = Path(os.environ.get("BER_FEAT_DIR", WORK_DIR / "feat"))


def pool_of(split: str, cols, country: str = None) -> pl.DataFrame:
    return pl.concat([scan_norm(split, 2, cols, country).with_columns(pl.lit(2, pl.Int8).alias("src")),
                      scan_norm(split, 3, cols, country).with_columns(pl.lit(3, pl.Int8).alias("src"))])


def s1_subset(split: str) -> dict:
    """Test: every S1. Train: WHOLE states (all S1 of a state), so that competition between S1 records for the
    same S2/S3 record is complete inside the subset, exactly as on test. States are shuffled per country and
    assigned to val until ~N_VAL_S1 * share, then to train until ~N_TRAIN_S1 * share."""
    if split == "test":
        return {"test": scan_norm(split, 1, ["entity_id"])["entity_id"]}
    if EXPLICIT_STATES:                     # dense, test-like states chosen explicitly
        s1 = scan_norm(split, 1, ["entity_id", "state"]).with_columns(pl.col("state").replace(STATE_GROUP))
        out = {k: s1.filter(pl.col("state").is_in(v))["entity_id"] for k, v in EXPLICIT_STATES.items()}
        print({k: (v, len(out[k])) for k, v in EXPLICIT_STATES.items()}, flush=True)
        return out
    s1 = scan_norm(split, 1, ["entity_id", "country", "state"]).with_columns(
        pl.col("state").replace(STATE_GROUP).alias("state"))
    share = s1.group_by("country").len().with_columns(pl.col("len") / pl.col("len").sum()).sort("country")
    rng = np.random.default_rng(SEED)
    val_ids, train_ids = [], []
    for country, frac in share.iter_rows():
        sizes = (s1.filter(pl.col("country") == country).group_by("state").len()
                   .sort("state").sample(fraction=1.0, shuffle=True, seed=int(rng.integers(1e9))))
        want_val, want_tr = N_VAL_S1 * frac, N_TRAIN_S1 * frac
        cap = want_val * (0.6 if len(sizes) >= 40 else 1.5)
        n_val = n_tr = 0
        v_states, t_states = [], []
        for st, n in sizes.iter_rows():
            if n_val < want_val:
                if n > cap:                 # val: only small/medium states -> several states in val
                    continue
                v_states.append(st); n_val += n
            elif n_tr < want_tr and n <= TRAIN_STATE_CAP:
                t_states.append(st); n_tr += n
        print(f"  {country}: val states {v_states} ({n_val:,}), train states {t_states} ({n_tr:,})", flush=True)
        c = s1.filter(pl.col("country") == country)
        val_ids.append(c.filter(pl.col("state").is_in(v_states))["entity_id"])
        train_ids.append(c.filter(pl.col("state").is_in(t_states))["entity_id"])
    return {"train": pl.concat(train_ids), "val": pl.concat(val_ids)}


def countries(split: str):
    cs = scan_norm(split, 1, ["country"])["country"].unique().sort().to_list()
    only = os.environ.get("BER_COUNTRIES")           # e.g. "US": build just these shards (parallel jobs)
    return [c for c in cs if c in only.split(",")] if only else cs


def build_candidates(split: str):
    """Per country (memory-lean, resumable): shards in CAND/<split>_<country>.parquet, then split by subset."""
    CAND.mkdir(parents=True, exist_ok=True)
    subsets = s1_subset(split)
    wanted = set(pl.concat(list(subsets.values())).to_list())
    shards = []
    for country in countries(split):
        shard = CAND / f"{split}_{country}.parquet"
        shards.append(shard)
        if shard.exists():
            print(f"[{split}] {country}: cached", flush=True)
            continue
        s1 = scan_norm(split, 1, BLOCK_COLS, country)
        pool = pool_of(split, BLOCK_COLS, country)
        keep = np.array([e in wanted for e in s1["entity_id"].to_list()])
        print(f"[{split}] {country}: {keep.sum():,} query S1 / {len(s1):,} S1 vs {len(pool):,} pool", flush=True)
        t0 = time.time()
        u = run_country(s1, pool, None, keep, log=lambda m: print(m, flush=True))
        print(f"  {country}: {len(u):,} pairs in {time.time() - t0:.0f}s", flush=True)
        u = u.with_columns(pl.Series("s1", s1["entity_id"].to_numpy()[u["q"].to_numpy()]),
                           pl.Series("cand", pool["entity_id"].to_numpy()[u["c"].to_numpy()]))
        u.drop("q", "c").write_parquet(shard)
        del s1, pool, u
        gc.collect()
    if os.environ.get("BER_COUNTRIES"):                # partial (parallel) job: the full run merges
        return
    cand = pl.concat([pl.read_parquet(f) for f in shards], how="diagonal")
    for name, ids in subsets.items():
        part = cand.filter(pl.col("s1").is_in(ids.implode()))
        part.write_parquet(CAND / f"{name}.parquet")
        print(f"  {name}: {len(part):,} pairs, {len(part) / len(ids):.1f}/S1", flush=True)
        if split == "train":
            gt = load_ground_truth().drop_nulls().filter(pl.col("source1_entity_id").is_in(ids.implode()))
            hit = part.join(gt.rename({"source1_entity_id": "s1", "match": "cand"}), on=["s1", "cand"]).height
            print(f"  {name}: blocking recall {hit / len(gt):.4f} ({hit:,}/{len(gt):,})", flush=True)
            for c in [c for c in part.columns if c.startswith("sim_")]:
                h = part.filter(pl.col(c).is_not_null()).join(
                    gt.rename({"source1_entity_id": "s1", "match": "cand"}), on=["s1", "cand"]).height
                print(f"    {c:20s} recall {h / len(gt):.4f}", flush=True)


def build_features(name: str, split: str, chunk: int = 250_000, max_new: int = 12) -> bool:
    """Resumable: existing shards are skipped and at most `max_new` chunks are computed per process
    (the caller re-invokes until it returns True), so memory never accumulates in one long process.
    Each chunk is written as a shard; shards are concatenated lazily at the end."""
    FEAT.mkdir(parents=True, exist_ok=True)
    cand_lf = pl.scan_parquet(CAND / f"{name}.parquet")            # read chunk by chunk, never whole
    n_cand = cand_lf.select(pl.len()).collect().item()
    cols = ["entity_id", "is_domain"] + STR_COLS
    lazy = lambda src, ids: (pl.scan_parquet(WORK_DIR / "norm" / f"{split}_source{src}.parquet")
                               .select(cols).filter(pl.col("entity_id").is_in(ids)).collect())
    # name genericity: how many S1 records share the same sorted-token name
    s1_freq = scan_norm(split, 1, ["name_sorted"]).group_by("name_sorted").len("n_s1_same_name")
    shard_dir = FEAT / f"{name}_shards"
    shard_dir.mkdir(exist_ok=True)
    gt = None
    if split == "train":
        gt = load_ground_truth().drop_nulls().rename({"source1_entity_id": "s1", "match": "cand"})                                 .with_columns(pl.lit(1, pl.Int8).alias("y"))
    t0 = time.time()
    done_new = 0
    # BER_FEAT_SHARD="k/n": this process only builds chunks i % n == k (n processes in parallel on a big node)
    k, n = map(int, os.environ.get("BER_FEAT_SHARD", "0/1").split("/"))
    for i, s in enumerate(range(0, n_cand, chunk)):
        if i % n != k or (shard_dir / f"{i:04d}.parquet").exists():
            continue
        if done_new >= max_new:
            return False
        done_new += 1
        c = cand_lf.slice(s, chunk).collect()
        # load only the records this chunk needs (filtered while scanning -> small memory, cheap restarts)
        s1 = lazy(1, c["s1"].unique().implode())
        c_ids = c["cand"].unique().implode()
        pool = pl.concat([lazy(2, c_ids).with_columns(pl.lit(2, pl.Int8).alias("src")),
                          lazy(3, c_ids).with_columns(pl.lit(3, pl.Int8).alias("src"))])
        L = c.select(pl.col("s1").alias("entity_id")).join(s1, on="entity_id", how="left", maintain_order="left")
        R = c.select(pl.col("cand").alias("entity_id")).join(pool, on="entity_id", how="left",
                                                             maintain_order="left")
        F = pair_features(L, R)
        extra = L.select("name_sorted").join(s1_freq, on="name_sorted", how="left", maintain_order="left")
        F = F.with_columns(extra["n_s1_same_name"].fill_null(0).cast(pl.Int32), R["src"])
        part = pl.concat([c, F], how="horizontal")
        if gt is not None:
            part = part.join(gt, on=["s1", "cand"], how="left").with_columns(pl.col("y").fill_null(0))
        part.write_parquet(shard_dir / f"{i:04d}.parquet")
        print(f"  features {name}: {s + len(c):,}/{n_cand:,} ({time.time() - t0:.0f}s)", flush=True)
    if len(list(shard_dir.glob("*.parquet"))) < -(-n_cand // chunk):
        return True                          # other shard processes still running; the final call merges
    pl.scan_parquet(shard_dir / "*.parquet").sink_parquet(FEAT / f"{name}.parquet")
    return True


def feature_cols(df):
    names = df.columns if hasattr(df, "columns") else list(df.keys())
    return [c for c in names if c not in ("s1", "cand", "y")]


def fit_done(name: str) -> bool:
    """Marker newer than the train features -> this model is already trained on them (parallel early job)."""
    m = WORK_DIR / f"{name}.done"
    return m.exists() and m.stat().st_mtime > (FEAT / "train.parquet").stat().st_mtime


def fit():
    if fit_done("fit_s1"):
        print("stage-1 already trained on these features: skip", flush=True)
        return
    cols = feature_cols(pl.read_parquet_schema(FEAT / "train.parquet"))
    tr = pl.read_parquet(FEAT / "train.parquet", columns=cols + ["y"])
    Xtr, ytr = tr.select(cols).cast(pl.Float32).to_numpy(), tr["y"].to_numpy()
    del tr                                    # free the table before LightGBM bins the matrix (RAM)
    gc.collect()
    va = pl.read_parquet(FEAT / "val.parquet")
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=127, num_threads=N_JOBS, verbose=-1, seed=SEED)
    dtr = lgb.Dataset(Xtr, ytr, feature_name=cols, free_raw_data=True)
    dva = lgb.Dataset(va.select(cols).cast(pl.Float32).to_numpy(), va["y"].to_numpy(), reference=dtr)
    m = lgb.train(params, dtr, 2000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    m.save_model(str(WORK_DIR / "lgb_v1.txt"))
    va = va.with_columns(pl.Series("p", m.predict(va.select(cols).cast(pl.Float32).to_numpy())))
    va.select("s1", "cand", "p", "y").write_parquet(WORK_DIR / "val_scored.parquet")
    truth = load_ground_truth().rename({"source1_entity_id": "s1", "match": "cand"})
    val_ids = s1_subset("train")["val"]
    truth = truth.filter(pl.col("s1").is_in(val_ids.implode()))
    ceiling = va.filter(pl.col("y") == 1).height / truth.drop_nulls().height
    grid = np.round(np.arange(0.2, 0.97, 0.02), 2)
    (thr, f), res = sweep(va.select("s1", "cand", "p"), truth, val_ids, grid)
    print(f"blocking recall on val: {ceiling:.4f}")
    print("threshold sweep:", [(t, round(s, 4)) for t, s in res])
    print(f"BEST val macro F0.5 = {f:.4f} at threshold {thr}")
    imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(c, int(g)) for c, g in imp[:15]])
    json.dump({"threshold": float(thr), "val_f05": f, "cols": cols, "iters": m.best_iteration},
              open(WORK_DIR / "lgb_v1.json", "w"))
    (WORK_DIR / "fit_s1.done").touch()


def predict(keep_min: float = 0.01):
    """Score test feature shards one at a time (72M pairs do not fit in RAM), keep pairs with p >= keep_min
    (lower-scored pairs can never be selected), then one-to-one + threshold. candidate_pairs.tsv is written
    country by country from the candidate shards (= exactly the pairs the model scored)."""
    meta = json.load(open(WORK_DIR / "lgb_v1.json"))
    m = lgb.Booster(model_file=str(WORK_DIR / "lgb_v1.txt"))
    kept = []
    shards = sorted((FEAT / "test_shards").glob("*.parquet"))
    t0 = time.time()
    for i, f in enumerate(shards):
        te = pl.read_parquet(f, columns=["s1", "cand"] + meta["cols"])
        p = m.predict(te.select(meta["cols"]).cast(pl.Float32).to_numpy(), num_threads=N_JOBS)
        kept.append(te.select("s1", "cand").with_columns(pl.Series("p", p.astype(np.float32)))
                      .filter(pl.col("p") >= keep_min))
        if i % 10 == 0:
            print(f"  scored shard {i + 1}/{len(shards)} ({time.time() - t0:.0f}s)", flush=True)
    te = pl.concat(kept)
    te.write_parquet(WORK_DIR / "test_scored.parquet")
    s1_ids = scan_norm("test", 1, ["entity_id"])["entity_id"].to_list()
    final = one_to_one(te).filter(pl.col("p") >= meta["threshold"])
    out = write_lists(s1_ids, final.select("s1", "cand"), OUT_DIR / "matching_results.tsv", "matched_entity_ids")
    n = out["matched_entity_ids"].str.len_chars()
    print(f"wrote matching_results: {len(out):,} rows; empty lists {(n == 0).mean():.3f}; "
          f"mean matches {final.height / len(out):.2f}", flush=True)
    write_candidate_file(s1_ids)


def write_candidate_file(s1_ids):
    path = OUT_DIR / "candidate_pairs.tsv"
    seen = set()
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for shard in sorted(CAND.glob("test_*.parquet")):
            agg = (pl.read_parquet(shard, columns=["s1", "cand"])
                     .group_by("s1").agg(pl.col("cand").unique().sort().str.join(",")))
            for s1, c in agg.iter_rows():
                fh.write(f"{s1}\t{c}\n")
                seen.add(s1)
            del agg
        for s1 in s1_ids:                       # S1 with no candidates at all
            if s1 not in seen:
                fh.write(f"{s1}\t\n")
    print(f"wrote candidate_pairs: {len(seen):,} S1 with candidates", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "candidates":
        build_candidates(sys.argv[2])
    elif cmd == "features":            # exit code 3 = more chunks to do (re-invoke)
        if sys.argv[2] == "train":
            ok = build_features("train", "train") and build_features("val", "train")
        else:
            ok = build_features("test", "test")
        sys.exit(0 if ok else 3)
    elif cmd == "features_named":      # e.g. dense-leg pairs: features_named train_new train
        sys.exit(0 if build_features(sys.argv[2], sys.argv[3]) else 3)
    elif cmd == "fit":
        fit()
    elif cmd == "predict":
        predict()
