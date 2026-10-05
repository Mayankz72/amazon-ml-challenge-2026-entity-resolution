"""GPU dense blocking leg: fine-tuned multilingual-e5-small (MIT) embeddings of "name | address", kNN per country.

Adds true pairs that the TF-IDF / key legs miss (transliterations, reordered or heavily noised text).

  python dense_block.py finetune             # contrastive (InfoNCE, in-batch negatives) on train matches,
                                             #   validation S1 (BER_CAND_DIR/val.parquet) left out
  python dense_block.py embed <split> <src>  # -> WORK/dense/<split>_s<src>.npy (+ ids parquet)
  python dense_block.py search <split>       # -> WORK/dense/<split>_pairs.parquet (s1, cand, sim_dense, rk_dense)
  python dense_block.py recall               # val: recall of TF-IDF/key candidates vs + dense leg
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from blocking import STATE_GROUP
from config import SEED, WORK_DIR
from data import load_ground_truth

BASE = os.environ.get("DENSE_BASE", "intfloat/multilingual-e5-small")
DENSE = WORK_DIR / os.environ.get("DENSE_TAG", "dense")      # one folder per embedding variant
MODEL_DIR = DENSE / "model"
CAND = Path(os.environ.get("BER_CAND_DIR", WORK_DIR / "cand_v4"))
MAX_LEN = 64
K_FWD = int(os.environ.get("DENSE_K", 20))       # S1 -> pool neighbours
K_REV = int(os.environ.get("DENSE_K_REV", 3))    # pool -> S1 neighbours (keeps pairs where the pool record's best S1)
TRAIN_STATES = os.environ.get("DENSE_STATES", "va,tn,kl,il,ka").split(",")   # train split: train + val states


def texts(df: pl.DataFrame) -> list:
    return df.select(pl.concat_str([pl.lit("query: "), pl.col("business_name").fill_null(""), pl.lit(" | "),
                                    pl.col("business_address").fill_null("")]))[:, 0].to_list()


def records(split: str, src: int) -> pl.DataFrame:
    lf = pl.scan_parquet(WORK_DIR / "norm" / f"{split}_source{src}.parquet").select(
        "entity_id", "business_name", "business_address", "country", "state")
    if split == "train":
        inside = pl.col("state").replace(STATE_GROUP).is_in(TRAIN_STATES)
        if os.environ.get("DENSE_STATES_INVERT") == "1":      # the states NOT listed (CE training on unused states)
            inside = ~inside
        lf = lf.filter(inside | pl.col("state").is_null() | (pl.col("state") == ""))
    return lf.collect()


def _encode(model, tok, txt, bs=1024):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(txt), bs):
            enc = tok(txt[i:i + bs], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)
            out.append(e.half().cpu().numpy())
    return np.concatenate(out)


def finetune(n_pairs: int = 600_000, bs: int = 256, lr: float = 3e-5, tau: float = 0.05):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(SEED)
    if os.environ.get("DENSE_EXCLUDE_STATES") == "1":   # no S1 of any train/val state (no in-sample sims)
        st = pl.scan_parquet(WORK_DIR / "norm" / "train_source1.parquet").select("entity_id", "state").collect()
        held = st.filter(pl.col("state").replace(STATE_GROUP).is_in(TRAIN_STATES))["entity_id"].implode()
    else:
        held = pl.read_parquet(CAND / "val.parquet", columns=["s1"])["s1"].unique().implode()
    gt = load_ground_truth().drop_nulls().filter(~pl.col("source1_entity_id").is_in(held))
    gt = gt.sample(min(n_pairs, len(gt)), seed=SEED)
    s1 = pl.scan_parquet(WORK_DIR / "raw" / "train_source1.parquet").filter(
        pl.col("entity_id").is_in(gt["source1_entity_id"].unique().implode())).collect()
    pool = pl.concat([pl.scan_parquet(WORK_DIR / "raw" / f"train_source{s}.parquet").filter(
        pl.col("entity_id").is_in(gt["match"].unique().implode())).collect() for s in (2, 3)])
    ta = dict(zip(s1["entity_id"].to_list(), texts(s1)))
    tb = dict(zip(pool["entity_id"].to_list(), texts(pool)))
    A = [ta[x] for x in gt["source1_entity_id"].to_list()]
    B = [tb[x] for x in gt["match"].to_list()]
    print(f"finetune {BASE}: {len(A):,} pairs", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    perm = np.random.default_rng(SEED).permutation(len(A))
    steps = len(A) // bs
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.05 * steps)) * max(0.0, 1 - s / steps))

    def emb(txt):
        enc = tok(txt, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to("cuda")
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)
    model.train()
    t0 = time.time()
    for st in range(steps):
        idx = perm[st * bs:(st + 1) * bs]
        with torch.autocast("cuda", dtype=torch.float16):
            ea, eb = emb([A[i] for i in idx]), emb([B[i] for i in idx])
            logits = (ea @ eb.T).float() / tau
        lab = torch.arange(len(idx), device="cuda")
        loss = (torch.nn.functional.cross_entropy(logits, lab) + torch.nn.functional.cross_entropy(logits.T, lab)) / 2
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        if st % 200 == 0:
            print(f"step {st}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR); tok.save_pretrained(MODEL_DIR)
    print("saved", MODEL_DIR, flush=True)


def embed(split: str, src: int):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR).cuda().eval()
    df = records(split, src)
    # sort by text length -> less padding, ~2x faster
    df = df.with_columns((pl.col("business_name").str.len_chars() + pl.col("business_address").str.len_chars())
                         .fill_null(0).alias("_l")).sort("_l").drop("_l")
    t0 = time.time()
    E = _encode(model, tok, texts(df))
    DENSE.mkdir(parents=True, exist_ok=True)
    np.save(DENSE / f"{split}_s{src}.npy", E)
    df.select("entity_id", "country").write_parquet(DENSE / f"{split}_s{src}_ids.parquet")
    print(f"embedded {split} s{src}: {len(df):,} in {time.time() - t0:.0f}s", flush=True)


def search(split: str):
    import torch
    ids = {s: pl.read_parquet(DENSE / f"{split}_s{s}_ids.parquet") for s in (1, 2, 3)}
    emb = {s: np.load(DENSE / f"{split}_s{s}.npy") for s in (1, 2, 3)}
    out = []
    for country in ids[1]["country"].unique().sort().to_list():
        i1 = np.flatnonzero((ids[1]["country"] == country).to_numpy())
        P_ids = pl.concat([ids[2]["entity_id"].filter(ids[2]["country"] == country),
                           ids[3]["entity_id"].filter(ids[3]["country"] == country)])
        P = np.concatenate([emb[2][(ids[2]["country"] == country).to_numpy()],
                            emb[3][(ids[3]["country"] == country).to_numpy()]])
        if len(i1) == 0 or len(P) == 0:
            continue
        Q = torch.from_numpy(emb[1][i1]).cuda(); Pt = torch.from_numpy(P).cuda()
        q_ids = ids[1]["entity_id"].gather(i1)
        t0 = time.time()
        rows_s1, rows_c, rows_sim = [], [], []
        qc = max(64, int(3e9 // (2 * len(Pt))))                        # score block <= ~3 GB on the GPU
        for a in range(0, len(Q), qc):                                     # forward: S1 -> top-K pool
            s, j = torch.topk(Q[a:a + qc] @ Pt.T, K_FWD, dim=1)
            rows_s1.append(np.repeat(np.arange(a, a + len(s)), K_FWD)); rows_c.append(j.flatten().cpu().numpy())
            rows_sim.append(s.flatten().float().cpu().numpy())
        pc = max(64, int(3e9 // (2 * len(Q))))
        for a in range(0, len(Pt), pc):                                    # reverse: pool -> top-K_REV S1
            s, j = torch.topk(Pt[a:a + pc] @ Q.T, K_REV, dim=1)
            rows_s1.append(j.flatten().cpu().numpy()); rows_c.append(np.repeat(np.arange(a, a + len(s)), K_REV))
            rows_sim.append(s.flatten().float().cpu().numpy())
        d = pl.DataFrame({"s1": q_ids.gather(np.concatenate(rows_s1)), "cand": P_ids.gather(np.concatenate(rows_c)),
                          "sim_dense": np.concatenate(rows_sim).astype(np.float32)})
        d = d.group_by("s1", "cand").agg(pl.col("sim_dense").max())
        d = d.with_columns(pl.col("sim_dense").rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias("rk_dense"))
        out.append(d)
        print(f"  {country}: {len(i1):,} S1 x {len(P):,} pool -> {len(d):,} pairs ({time.time() - t0:.0f}s)", flush=True)
        del Q, Pt; torch.cuda.empty_cache()
    pl.concat(out).write_parquet(DENSE / f"{split}_pairs.parquet")


def recall():
    gt = load_ground_truth().drop_nulls().rename({"source1_entity_id": "s1", "match": "cand"})
    dense = pl.read_parquet(DENSE / "train_pairs.parquet")
    for name in ("val", "train"):
        c = pl.read_parquet(CAND / f"{name}.parquet", columns=["s1", "cand"])
        g = gt.filter(pl.col("s1").is_in(c["s1"].unique().implode()))
        d = dense.filter(pl.col("s1").is_in(c["s1"].unique().implode()))
        hit_c = g.join(c, on=["s1", "cand"], how="semi").height
        hit_d = g.join(d.select("s1", "cand"), on=["s1", "cand"], how="semi").height
        u = pl.concat([c, d.select("s1", "cand")]).unique()
        hit_u = g.join(u, on=["s1", "cand"], how="semi").height
        n1 = c["s1"].n_unique()
        print(f"{name}: GT {len(g):,} | v4 recall {hit_c / len(g):.4f} ({len(c) / n1:.1f}/S1) | dense {hit_d / len(g):.4f} "
              f"({len(d) / n1:.1f}/S1) | union {hit_u / len(g):.4f} ({len(u) / n1:.1f}/S1)", flush=True)
        for k in (5, 10, 20):
            dk = d.filter(pl.col("rk_dense") <= k).select("s1", "cand")
            hk = g.join(pl.concat([c, dk]).unique(), on=["s1", "cand"], how="semi").height
            print(f"   union with dense rank<={k}: {hk / len(g):.4f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "finetune":
        finetune()
    elif cmd == "embed":
        embed(sys.argv[2], int(sys.argv[3]))
    elif cmd == "search":
        search(sys.argv[2])
    elif cmd == "recall":
        recall()
