"""Stage 2: context features computed over the whole scored candidate graph.

Input: pairs (s1, cand, p) where p is the stage-1 probability (out-of-fold on training data). Competition is
only meaningful when every S1 that could claim a record is present (test: all S1; train/val: whole states).
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

CTX_STR = ["name_core", "addr_clean", "house", "addr_nums"]


def context_features(pairs: pl.DataFrame, cand_attrs: pl.DataFrame) -> pl.DataFrame:
    """pairs: s1, cand, p (+ anything). cand_attrs: entity_id + CTX_STR for candidate records."""
    p = pl.col("p")
    df = pairs.with_columns(
        # --- within S1 (how does this candidate compare with the S1's other candidates)
        p.rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias("c_rank_s1"),
        p.max().over("s1").alias("c_pmax_s1"),
        (p.max().over("s1") - p).alias("c_gap_s1"),
        (p >= 0.5).sum().over("s1").cast(pl.Int16).alias("c_n05_s1"),
        (p >= 0.2).sum().over("s1").cast(pl.Int16).alias("c_n02_s1"),
        p.sum().over("s1").alias("c_psum_s1"),
        pl.len().over("s1").cast(pl.Int16).alias("c_ncand_s1"),
        # --- across S1 (competition for the same S2/S3 record)
        pl.len().over("cand").cast(pl.Int16).alias("c_nclaim"),
        p.max().over("cand").alias("c_pmax_cand"),
        p.rank("ordinal", descending=True).over("cand").cast(pl.Int16).alias("c_rank_cand"),
        (p.sum().over("cand") - p).alias("c_pother_cand"),
    )
    # margin to the best OTHER S1 claiming this record
    second = (df.select("cand", "p").sort("p", descending=True)
                .group_by("cand", maintain_order=True).agg(pl.col("p").slice(1, 1).first().alias("_p2")))
    df = df.join(second, on="cand", how="left").with_columns(
        pl.when(pl.col("c_rank_cand") == 1).then(p - pl.col("_p2").fill_null(0.0))
          .otherwise(p - pl.col("c_pmax_cand")).alias("c_margin_cand")).drop("_p2")
    # --- agreement with the S1's strongest siblings (duplicates of one business resemble each other):
    #     compare every candidate with the S1's top-3 candidates and keep the best / mean similarity
    attrs = cand_attrs.select("entity_id", *CTX_STR)
    me = df.select(pl.col("cand").alias("entity_id")).join(attrs, on="entity_id", how="left", maintain_order="left")
    mn, ma = me["name_core"].fill_null("").to_list(), me["addr_clean"].fill_null("").to_list()
    mh = me["house"].fill_null("").to_numpy()
    n_best = np.full(len(df), np.nan, np.float32); a_best = n_best.copy()
    n_sum = np.zeros(len(df), np.float32); a_sum = n_sum.copy(); cnt = np.zeros(len(df), np.float32)
    h_agree = np.zeros(len(df), np.int8); w_best = n_best.copy()
    for r in (1, 2, 3):
        top = df.filter(pl.col("c_rank_s1") == r).select("s1", pl.col("cand").alias("_sib"), pl.col("p").alias("_psib"))
        j = df.select("s1", "cand").join(top, on="s1", how="left", maintain_order="left")
        sib = j.select(pl.col("_sib").alias("entity_id")).join(attrs, on="entity_id", how="left", maintain_order="left")
        valid = (j["_sib"].is_not_null() & (j["_sib"] != j["cand"])).to_numpy()
        ns = cpdist(mn, sib["name_core"].fill_null("").to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        as_ = cpdist(ma, sib["addr_clean"].fill_null("").to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        sh = sib["house"].fill_null("").to_numpy()
        ps = j["_psib"].fill_null(0.0).to_numpy().astype(np.float32)
        ns[~valid] = np.nan; as_[~valid] = np.nan
        n_best = np.fmax(n_best, ns); a_best = np.fmax(a_best, as_)
        w_best = np.fmax(w_best, np.where(valid, ps * (ns + as_) / 200.0, np.nan))   # sibling strength x similarity
        n_sum += np.where(valid, ns, 0); a_sum += np.where(valid, as_, 0); cnt += valid
        h_agree += (valid & (mh != "") & (mh == sh)).astype(np.int8)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_mean, a_mean = n_sum / cnt, a_sum / cnt
    return df.with_columns(pl.Series("c_sib_name_max", n_best), pl.Series("c_sib_addr_max", a_best),
                           pl.Series("c_sib_name_mean", n_mean.astype(np.float32)),
                           pl.Series("c_sib_addr_mean", a_mean.astype(np.float32)),
                           pl.Series("c_sib_house_agree", h_agree), pl.Series("c_sib_weighted", w_best))


def cluster_features(df: pl.DataFrame, cand_attrs: pl.DataFrame, p_min: float = 0.1, top: int = 8,
                     chunk_s1: int = 150_000) -> pl.DataFrame:
    """Group structure inside each S1's candidate list: true matches form one tight group of near-copies,
    look-alike twins (orphan clusters) form another. For each candidate: size / strength of its own group and
    strength of the best competing group. Computed over the S1's top-`top` candidates with p >= p_min."""
    attrs = cand_attrs.select("entity_id", "name_core", "addr_clean", "house")
    base = (df.filter(pl.col("p") >= p_min).filter(pl.col("c_rank_s1") <= top)
              .select("s1", "cand", "p"))
    s1s = base["s1"].unique()
    outs = []
    for i in range(0, len(s1s), chunk_s1):
        b = base.filter(pl.col("s1").is_in(s1s.slice(i, chunk_s1).implode()))
        b = b.join(attrs.rename({"entity_id": "cand"}), on="cand", how="left").fill_null("")
        pr = b.join(b, on="s1", suffix="_d").filter(pl.col("cand") != pl.col("cand_d"))
        if len(pr) == 0:
            continue
        ns = cpdist(pr["name_core"].to_list(), pr["name_core_d"].to_list(), scorer=fuzz.token_set_ratio,
                    workers=-1, dtype=np.float32)
        as_ = cpdist(pr["addr_clean"].to_list(), pr["addr_clean_d"].to_list(), scorer=fuzz.token_set_ratio,
                     workers=-1, dtype=np.float32)
        same_h = (pr["house"] != "") & (pr["house"] == pr["house_d"])
        link = pl.Series(((ns >= 85) & (as_ >= 80)) | ((ns >= 75) & same_h.to_numpy() & (as_ >= 60)))
        pr = pr.select("s1", "cand", "p_d").with_columns(link.alias("_link"),
                                                         pl.Series("_ns", ns), pl.Series("_as", as_))
        agg = pr.group_by("s1", "cand").agg(
            pl.col("_link").sum().cast(pl.Int16).alias("g_size"),
            pl.col("p_d").filter(pl.col("_link")).mean().alias("g_pmean"),
            pl.col("p_d").filter(pl.col("_link")).max().alias("g_pmax"),
            pl.col("p_d").filter(~pl.col("_link")).max().alias("g_rival_pmax"),
            pl.col("_ns").max().alias("g_name_max"), pl.col("_as").max().alias("g_addr_max"),
            (pl.col("p_d") * pl.col("_link").cast(pl.Float32)).sum().alias("g_psum"))
        outs.append(agg)
    g = pl.concat(outs) if outs else pl.DataFrame(schema={"s1": pl.Utf8, "cand": pl.Utf8})
    return df.join(g, on=["s1", "cand"], how="left")


GROUP_COLS = ["g_size", "g_pmean", "g_pmax", "g_rival_pmax", "g_name_max", "g_addr_max", "g_psum"]


CTX_COLS = ["c_rank_s1", "c_pmax_s1", "c_gap_s1", "c_n05_s1", "c_n02_s1", "c_psum_s1", "c_ncand_s1",
            "c_nclaim", "c_pmax_cand", "c_rank_cand", "c_pother_cand", "c_margin_cand",
            "c_sib_name_max", "c_sib_addr_max", "c_sib_name_mean", "c_sib_addr_mean", "c_sib_house_agree",
            "c_sib_weighted"]


# ---------------------------------------------------------------------------------- decision rule
def expected_f05_select(pairs: pl.DataFrame, score: str = "p", min_p: float = 0.05, beta2: float = 0.25):
    """Per S1: sort candidates by calibrated p and choose k maximising E[F0.5] (ratio-of-expectations
    approximation: E[F] ≈ (1+b²)·Σ_top-k p / (b²·E|true| + k)); k = 0 scores P(no match) ≈ Π(1-p)."""
    d = (pairs.filter(pl.col(score) >= min_p).sort(["s1", score], descending=[False, True])
              .with_columns(pl.col(score).cum_sum().over("s1").alias("_cs"),
                            pl.int_range(1, pl.len() + 1).over("s1").alias("_k"),
                            pl.col(score).sum().over("s1").alias("_et"),
                            (1 - pl.col(score)).log().sum().over("s1").exp().alias("_p0")))
    d = d.with_columns(((1 + beta2) * pl.col("_cs") / (beta2 * pl.col("_et") + pl.col("_k"))).alias("_ef"))
    best = d.group_by("s1").agg(pl.col("_ef").max().alias("_best"), pl.col("_p0").first())
    d = d.join(best, on="s1").with_columns(
        (pl.col("_ef") == pl.col("_best")).cast(pl.Int8).alias("_isbest"))
    kbest = d.filter(pl.col("_isbest") == 1).group_by("s1").agg(pl.col("_k").min().alias("_kbest"),
                                                                 pl.col("_best").first(), pl.col("_p0").first())
    d = d.join(kbest.select("s1", "_kbest", pl.col("_best").alias("_bestv")), on="s1")
    chosen = d.filter((pl.col("_k") <= pl.col("_kbest")) & (pl.col("_bestv") > pl.col("_p0")))
    return chosen.select("s1", "cand")
