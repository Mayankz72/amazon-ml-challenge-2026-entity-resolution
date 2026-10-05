"""Stage 2: stage-1 probability + context features -> second LightGBM -> decision.

  python stage2.py fit       # OOF stage-1 on train (2 state folds), context features, stage-2 model, val sweep
  python stage2.py predict   # context features on test (stage-1 scores from pipeline.predict), final TSV
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from config import N_JOBS, OUT_DIR, SEED, WORK_DIR
from context import CTX_COLS, CTX_STR, GROUP_COLS, cluster_features, context_features, expected_f05_select
from edit_features import EDIT_STR, learn_llr, llr_features, raw_edits
from multiprocessing import Pool
from config import N_PROCS
from data import load_ground_truth
from decide import evaluate, one_to_one, sweep, write_lists
from pipeline import FEAT, s1_subset
from prep import scan_norm

P_MIN = 0.01          # context is computed over pairs with stage-1 p >= P_MIN (same rule on every split)
CE_SCORES = os.environ.get("S2_CE")   # optional parquet(s) s1, cand, ce: cross-encoder score as a stage-2 feature
S1_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                 max_bin=127, num_threads=N_JOBS, verbose=-1, seed=SEED)
S2_PARAMS = dict(S1_PARAMS, num_leaves=63, learning_rate=0.03)


def _X(df, cols):
    return df.select(cols).cast(pl.Float32).to_numpy()


def cand_attrs(split: str, ids: pl.Series) -> pl.DataFrame:
    """Normalised fields of the given S2/S3 records (lazy filter: never loads the whole pool)."""
    frames = [pl.scan_parquet(WORK_DIR / "norm" / f"{split}_source{s}.parquet")
                .select(["entity_id"] + CTX_STR).filter(pl.col("entity_id").is_in(ids.implode())).collect()
              for s in (2, 3)]
    return pl.concat(frames)


def oof_train(tr: pl.DataFrame, cols, iters: int) -> np.ndarray:
    """Out-of-fold stage-1 scores for the training pairs; folds are whole states (keeps competition intact)."""
    st = scan_norm("train", 1, ["entity_id", "state"]).rename({"entity_id": "s1"})
    tr_st = tr.select("s1").join(st, on="s1", how="left", maintain_order="left")["state"]
    states = sorted(tr_st.unique().to_list())
    rng = np.random.default_rng(SEED)
    fold_of = {s: int(rng.integers(2)) for s in states}
    fold = tr_st.replace_strict(fold_of, default=0).to_numpy()
    oof = np.zeros(len(tr), np.float32)
    for k in (0, 1):
        trn, hold = fold != k, fold == k
        d = lgb.Dataset(_X(tr.filter(pl.Series(trn)), cols), tr["y"].to_numpy()[trn])
        m = lgb.train(S1_PARAMS, d, iters)
        oof[hold] = m.predict(_X(tr.filter(pl.Series(hold)), cols))
        print(f"  OOF fold {k}: {hold.sum():,} pairs scored", flush=True)
    return oof


def s1_attrs(split: str, ids: pl.Series) -> pl.DataFrame:
    return (pl.scan_parquet(WORK_DIR / "norm" / f"{split}_source1.parquet").select(["entity_id"] + EDIT_STR)
              .filter(pl.col("entity_id").is_in(ids.implode())).collect())


def _edits_work(args):
    L, R = args
    return raw_edits(L, R)


def add_edits(df: pl.DataFrame, split: str, ca: pl.DataFrame, tables=None, fold=None):
    """Word/number edit evidence S1 vs candidate. Train: tables learned out-of-fold (fold array);
    val/test: given tables."""
    L = df.select(pl.col("s1").alias("entity_id")).join(s1_attrs(split, df["s1"].unique()), on="entity_id",
                                                        how="left", maintain_order="left").fill_null("")
    R = df.select(pl.col("cand").alias("entity_id")).join(ca, on="entity_id", how="left",
                                                          maintain_order="left").fill_null("")
    step = 100_000
    with Pool(N_PROCS) as pool:
        parts = pool.map(_edits_work, [(L.slice(i, step), R.slice(i, step)) for i in range(0, len(df), step)])
    ed = {k: [x for p in parts for x in p[0][k]] for k in parts[0][0]}
    num = {k: np.concatenate([np.array(p[1][k], np.int16) for p in parts]) for k in parts[0][1]}
    if tables is None:                           # out-of-fold on train
        y = df["y"].to_numpy()
        llr = {}
        for k in (0, 1):
            idx = np.where(fold == k)[0]
            tab = learn_llr({kk: [v for v, f in zip(vv, fold) if f != k] for kk, vv in ed.items()}, y[fold != k])
            part = llr_features({kk: [vv[i] for i in idx] for kk, vv in ed.items()}, tab)
            for kk, arr in part.items():
                llr.setdefault(kk, np.zeros(len(df), arr.dtype))[idx] = arr
        tables = learn_llr(ed, y)
    else:
        llr = llr_features(ed, tables)
    return df.with_columns(**num, **llr), tables, list(num) + list(llr)


def state_folds(s1_col: pl.Series) -> np.ndarray:
    st = scan_norm("train", 1, ["entity_id", "state"]).rename({"entity_id": "s1"})
    s = s1_col.to_frame("s1").join(st, on="s1", how="left", maintain_order="left")["state"]
    rng = np.random.default_rng(SEED)
    fold_of = {x: int(rng.integers(2)) for x in sorted(s.unique().to_list())}
    return s.replace_strict(fold_of, default=0).to_numpy()


RARITY_COLS = ["r_same_key", "r_cand_no_addr", "r_s1_key_n", "r_s1_of_cand_key_n", "r_pool_key_n",
               "r_pool_key_noaddr_n"]


def add_rarity(ctx: pl.DataFrame, split: str) -> pl.DataFrame:
    """Name-uniqueness evidence (key = sorted core-name tokens, counted per country over the whole split).
    Half of the missed matches have a name-only candidate: then the only evidence is how unique the name is."""
    key = pl.col("name_core").fill_null("").str.split(" ").list.sort().list.join(" ")
    rd = lambda s: pl.scan_parquet(WORK_DIR / "norm" / f"{split}_source{s}.parquet").select(
        "entity_id", "country", key.alias("k"), (pl.col("addr_clean").fill_null("") == "").alias("no_addr"))
    s1 = rd(1).collect()
    pool = pl.concat([rd(2), rd(3)]).collect()
    s1_n = s1.group_by("country", "k").len("n_s1")
    pool_n = pool.group_by("country", "k").agg(pl.len().alias("n_pool"), pl.col("no_addr").sum().alias("n_pool_na"))
    a = ctx.select("s1", "cand").join(s1.select(pl.col("entity_id").alias("s1"), pl.col("k").alias("k1"), "country"),
                                      on="s1", how="left", maintain_order="left") \
           .join(pool.select(pl.col("entity_id").alias("cand"), pl.col("k").alias("k2"), "no_addr"),
                 on="cand", how="left", maintain_order="left")
    a = a.join(s1_n.rename({"k": "k1", "n_s1": "r_s1_key_n"}), on=["country", "k1"], how="left", maintain_order="left") \
         .join(s1_n.rename({"k": "k2", "n_s1": "r_s1_of_cand_key_n"}), on=["country", "k2"], how="left", maintain_order="left") \
         .join(pool_n.rename({"k": "k2", "n_pool": "r_pool_key_n", "n_pool_na": "r_pool_key_noaddr_n"}),
               on=["country", "k2"], how="left", maintain_order="left")
    a = a.with_columns((pl.col("k1") == pl.col("k2")).cast(pl.Int8).alias("r_same_key"),
                       pl.col("no_addr").cast(pl.Int8).alias("r_cand_no_addr"))
    return pl.concat([ctx, a.select([pl.col(c).fill_null(0).cast(pl.Float32) for c in RARITY_COLS])], how="horizontal")


def build_ctx(df: pl.DataFrame, split: str, tables=None):
    keep = df.filter(pl.col("p") >= P_MIN)
    ca = cand_attrs(split, keep["cand"].unique())
    t0 = time.time()
    ctx = context_features(keep, ca)
    ctx = cluster_features(ctx, ca)
    fold = state_folds(ctx["s1"]) if tables is None else None
    ctx, tables, ecols = add_edits(ctx, split, ca, tables, fold)
    if os.environ.get("S2_RARITY") == "1":
        ctx = add_rarity(ctx, split)
        ecols = ecols + RARITY_COLS
    if CE_SCORES:
        ce = pl.concat([pl.read_parquet(f) for f in CE_SCORES.split(",")]).unique(["s1", "cand"])
        ctx = ctx.join(ce.select("s1", "cand", "ce"), on=["s1", "cand"], how="left")
        print(f"  ce coverage {split}: {ctx['ce'].is_not_null().mean():.4f}", flush=True)
        ecols = ecols + ["ce"]
    # several cross-encoders: S2_CE_EXTRA="tag=f1,f2;tag2=f3" -> column ce_<tag> each
    for spec in filter(None, os.environ.get("S2_CE_EXTRA", "").split(";")):
        tag, files = spec.split("=", 1)
        ce = pl.concat([pl.read_parquet(f) for f in files.split(",")]).unique(["s1", "cand"])
        ctx = ctx.join(ce.select("s1", "cand", pl.col("ce").alias(f"ce_{tag}")), on=["s1", "cand"], how="left")
        print(f"  ce_{tag} coverage {split}: {ctx[f'ce_{tag}'].is_not_null().mean():.4f}", flush=True)
        ecols = ecols + [f"ce_{tag}"]
    ce_cols = [c for c in ecols if c == "ce" or c.startswith("ce_")]
    if ce_cols and os.environ.get("S2_CE_CTX") == "1":
        # competition on the cross-encoder score: rank / gap to the best within the S1 and within the candidate
        m = pl.mean_horizontal(ce_cols)
        ctx = ctx.with_columns(m.alias("cx_mean")).with_columns(
            pl.col("cx_mean").rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("cx_rank_s1"),
            (pl.col("cx_mean").max().over("s1") - pl.col("cx_mean")).alias("cx_gap_s1"),
            (pl.col("cx_mean") > 0.5).sum().over("s1").cast(pl.Float32).alias("cx_n05_s1"),
            pl.col("cx_mean").rank("ordinal", descending=True).over("cand").cast(pl.Float32).alias("cx_rank_cand"),
            (pl.col("cx_mean").max().over("cand") - pl.col("cx_mean")).alias("cx_gap_cand"))
        ecols = ecols + ["cx_mean", "cx_rank_s1", "cx_gap_s1", "cx_n05_s1", "cx_rank_cand", "cx_gap_cand"]
    print(f"  context for {split}: {len(ctx):,} pairs ({time.time() - t0:.0f}s)", flush=True)
    return ctx, tables, ecols


def fit():
    from pipeline import fit_done
    if fit_done("fit_s2"):
        print("stage-2 already trained on these features: skip", flush=True)
        return
    meta = json.load(open(WORK_DIR / "lgb_v1.json"))
    cols, iters = meta["cols"], meta["iters"]
    tr = pl.read_parquet(FEAT / "train.parquet")
    va = pl.read_parquet(FEAT / "val.parquet")
    t0 = time.time()
    oof_cache = WORK_DIR / "s2_oof_train.npy"            # deterministic, reused by `pairs` and later fits
    if oof_cache.exists():
        oof = np.load(oof_cache)
    else:
        oof = oof_train(tr, cols, iters)
        np.save(oof_cache, oof)
    tr = tr.with_columns(pl.Series("p", oof))
    m1 = lgb.Booster(model_file=str(WORK_DIR / "lgb_v1.txt"))
    va = va.with_columns(pl.Series("p", m1.predict(_X(va, cols)).astype(np.float32)))
    print(f"stage-1 scores ready ({time.time() - t0:.0f}s)", flush=True)
    tr, tables, ecols = build_ctx(tr, "train")
    va, _, _ = build_ctx(va, "train", tables)
    json.dump(tables, open(WORK_DIR / "edit_llr_tables.json", "w"))
    cols2 = cols + ["p"] + CTX_COLS + GROUP_COLS + ecols
    dtr = lgb.Dataset(_X(tr, cols2), tr["y"].to_numpy(), feature_name=cols2)
    dva = lgb.Dataset(_X(va, cols2), va["y"].to_numpy(), reference=dtr)
    m2 = lgb.train(S2_PARAMS, dtr, 3000, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    m2.save_model(str(WORK_DIR / "lgb_s2.txt"))
    va = va.with_columns(pl.Series("p2", m2.predict(_X(va, cols2)).astype(np.float32)))
    va.select("s1", "cand", "p", "p2", "y").write_parquet(WORK_DIR / "val_scored_s2.parquet")

    val_ids = s1_subset("train")["val"]
    truth = load_ground_truth().rename({"source1_entity_id": "s1", "match": "cand"}) \
                               .filter(pl.col("s1").is_in(val_ids.implode()))
    grid = np.round(np.arange(0.3, 0.97, 0.02), 2)
    (t1, f1), _ = sweep(va.select("s1", "cand", "p"), truth, val_ids, grid, score="p")
    (t2, f2), res = sweep(va.select("s1", "cand", "p2"), truth, val_ids, grid, score="p2")
    ef = evaluate(expected_f05_select(one_to_one(va.select("s1", "cand", "p2"), "p2"), score="p2"),
                  truth, val_ids)
    print(f"stage-1 only : F0.5 {f1:.4f} @ {t1}")
    print(f"stage-2      : F0.5 {f2:.4f} @ {t2}")
    print(f"stage-2 + expected-F0.5 selection: F0.5 {ef:.4f}")
    print("stage-2 sweep:", [(t, round(s, 4)) for t, s in res])
    imp = sorted(zip(cols2, m2.feature_importance("gain")), key=lambda x: -x[1])
    print("top stage-2 features:", [(c, int(g)) for c, g in imp[:15]])
    rule = "expected" if ef > f2 else "threshold"
    json.dump({"threshold": float(t2), "val_f05": max(f2, ef), "rule": rule, "cols": cols2,
               "stage1_val_f05": f1}, open(WORK_DIR / "lgb_s2.json", "w"))
    (WORK_DIR / "fit_s2.done").touch()


def predict():
    meta = json.load(open(WORK_DIR / "lgb_s2.json"))
    m2 = lgb.Booster(model_file=str(WORK_DIR / "lgb_s2.txt"))
    scored = pl.read_parquet(WORK_DIR / "test_scored.parquet")          # s1, cand, p (stage 1, p >= 0.01)
    tables = json.load(open(WORK_DIR / "edit_llr_tables.json"))
    ctx, _, _ = build_ctx(scored, "test", tables)
    ctx = ctx.drop("p")
    # candidate_pairs.tsv = exactly the pairs the final (stage-2) model scores: blocking -> stage-1 learned
    # filter (p >= P_MIN) -> ~4 candidates per S1 (challenge rule: smaller candidate sets rank higher)
    s1_all = scan_norm("test", 1, ["entity_id"])["entity_id"].to_list()
    write_lists(s1_all, ctx.select("s1", "cand"), OUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids")
    print(f"candidate_pairs.tsv: {len(ctx):,} pairs = {len(ctx) / len(s1_all):.2f} per S1", flush=True)
    print(f"test context: {len(ctx):,} pairs", flush=True)
    s1_cols = [c for c in meta["cols"] if c not in ctx.columns or c == "p"]   # stage-1 features + p
    out = []
    for f in sorted((FEAT / "test_shards").glob("*.parquet")):
        sh = pl.read_parquet(f, columns=["s1", "cand"] + [c for c in s1_cols if c != "p"])
        sh = sh.join(scored, on=["s1", "cand"], how="inner").join(ctx, on=["s1", "cand"], how="inner")
        if len(sh):
            out.append(sh.select("s1", "cand", pl.Series("p2", m2.predict(_X(sh, meta["cols"])).astype(np.float32))))
    te = pl.concat(out)
    te.write_parquet(WORK_DIR / "test_scored_s2.parquet")
    o = one_to_one(te, "p2")
    final = expected_f05_select(o, score="p2") if meta["rule"] == "expected" else \
        o.filter(pl.col("p2") >= meta["threshold"]).select("s1", "cand")
    s1_ids = scan_norm("test", 1, ["entity_id"])["entity_id"].to_list()
    res = write_lists(s1_ids, final, OUT_DIR / "matching_results.tsv", "matched_entity_ids")
    n = res["matched_entity_ids"].str.len_chars()
    print(f"wrote {len(res):,} rows; empty {(n == 0).mean():.3f}; mean matches {final.height / len(res):.2f}")


def pairs():
    """Pair lists stage 2 will see (stage-1 p >= P_MIN): train (OOF), val, test -> for cross-encoder scoring."""
    meta = json.load(open(WORK_DIR / "lgb_v1.json"))
    cols, iters = meta["cols"], meta["iters"]
    oof_cache = WORK_DIR / "s2_oof_train.npy"
    tr = pl.read_parquet(FEAT / "train.parquet", columns=["s1", "cand", "y"] + cols)
    if not oof_cache.exists():
        np.save(oof_cache, oof_train(tr, cols, iters))
    tr = tr.select("s1", "cand", "y").with_columns(pl.Series("p", np.load(oof_cache)))
    m1 = lgb.Booster(model_file=str(WORK_DIR / "lgb_v1.txt"))
    va = pl.read_parquet(FEAT / "val.parquet", columns=["s1", "cand", "y"] + cols)
    va = va.select("s1", "cand", "y").with_columns(pl.Series("p", m1.predict(_X(va, cols)).astype(np.float32)))
    te = pl.read_parquet(WORK_DIR / "test_scored.parquet")
    for name, d in (("train", tr), ("val", va), ("test", te)):
        d.filter(pl.col("p") >= P_MIN).drop("p").write_parquet(WORK_DIR / f"s2_pairs_{name}.parquet")
        print(f"s2 pairs {name}: {d.filter(pl.col('p') >= P_MIN).height:,}", flush=True)


if __name__ == "__main__":
    {"fit": fit, "predict": predict, "pairs": pairs}[sys.argv[1]]()
