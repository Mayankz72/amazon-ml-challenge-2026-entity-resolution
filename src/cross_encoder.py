"""GPU cross-encoder: reads (S1 record, candidate record) jointly and scores P(same business).

Used only on the small stage-1-filtered candidate set (~5 per S1), so inference is affordable. Its score becomes
a stage-2 feature. Model: microsoft/mdeberta-v3-base (MIT licence, ~280M params, multilingual).

  python cross_encoder.py export            # (CPU) write pair texts: work/ce/{train,val,test}.parquet
  python cross_encoder.py train             # (GPU) fine-tune on train pairs, eval on val
  python cross_encoder.py score test        # (GPU) score pairs -> work/ce/test_scores.parquet
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

WORK = Path(os.environ.get("BER_WORK_DIR", Path(os.environ.get("AMZML_HOME", Path.home() / "amzml")) / "work"))
CE = WORK / "ce"
MODEL = os.environ.get("CE_MODEL", "microsoft/mdeberta-v3-base")
MAX_LEN = int(os.environ.get("CE_MAX_LEN", 96))
OUT = CE / os.environ.get("CE_TAG", "model")          # one folder per model variant
EXCLUDE = os.environ.get("CE_EXCLUDE_S1")             # parquet(s), comma-separated, column s1: kept out of CE training


def rec_text(name: str, addr: str) -> str:
    return f"{name} | {addr}"


def export():
    """Pair texts for the hard, stage-1-filtered candidate sets (raw text: the transformer reads the noise).
    train: v-train blocking pairs with stage-1 p >= 0.005 (in-sample p, used only as a selection filter);
    val / test: the exact stage-2 input sets (p >= 0.01)."""
    import json
    import lightgbm as lgb
    CE.mkdir(parents=True, exist_ok=True)
    raw = lambda split, s: pl.scan_parquet(WORK / "raw" / f"{split}_source{s}.parquet")
    meta = json.load(open(WORK / "lgb_v1.json"))
    m1 = lgb.Booster(model_file=str(WORK / "lgb_v1.txt"))
    tr = pl.read_parquet(WORK / "feat" / "train.parquet", columns=["s1", "cand", "y"] + meta["cols"])
    tr = tr.with_columns(pl.Series("p", m1.predict(tr.select(meta["cols"]).cast(pl.Float32).to_numpy())))            .filter(pl.col("p") >= 0.005).select("s1", "cand", "y")
    va = pl.read_parquet(WORK / "val_scored_s2.parquet", columns=["s1", "cand", "y", "p"]).filter(pl.col("p") >= 0.01)
    te = pl.read_parquet(WORK / "test_scored.parquet").filter(pl.col("p") >= 0.01)
    for name, split, pairs in (("train", "train", tr), ("val", "train", va.drop("p")), ("test", "test", te.drop("p"))):
        s1 = raw(split, 1).filter(pl.col("entity_id").is_in(pairs["s1"].unique().implode())).collect()
        pool = pl.concat([raw(split, s).filter(pl.col("entity_id").is_in(pairs["cand"].unique().implode())).collect()
                          for s in (2, 3)])
        txt = lambda df, col: df.select(pl.col("entity_id").alias(col), pl.concat_str(
            ["business_name", "business_address"], separator=" | ").alias("text_" + ("a" if col == "s1" else "b")))
        d = pairs.join(txt(s1, "s1"), on="s1", how="left").join(txt(pool, "cand"), on="cand", how="left")
        d.write_parquet(CE / f"{name}.parquet")
        print(f"exported {name}: {len(d):,} pairs" + (f", positives {int(d['y'].sum()):,}" if "y" in d.columns else ""),
              flush=True)


def _dataset(df, tok):
    import torch
    enc = tok(df["text_a"].to_list(), df["text_b"].to_list(), truncation=True, max_length=MAX_LEN,
              padding="max_length", return_tensors="pt")
    y = torch.tensor(df["y"].to_numpy() if "y" in df.columns else np.zeros(len(df)), dtype=torch.float32)
    return torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"], y)


def train(epochs: float = 1.0, bs: int = int(os.environ.get("CE_BS", 64)), lr: float = float(os.environ.get("CE_LR", 2e-5)),
          max_train: int = int(os.environ.get("CE_MAX_TRAIN", 3_000_000))):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).cuda()
    tr = pl.read_parquet(CE / os.environ.get("CE_TRAIN_FILE", "train.parquet"))
    if EXCLUDE:                                       # e.g. the validation states: no leak into stage-2 val
        ex = pl.concat([pl.read_parquet(f, columns=["s1"]) for f in EXCLUDE.split(",")])["s1"].unique()
        tr = tr.filter(~pl.col("s1").is_in(ex.implode()))
    print(f"CE {MODEL} -> {OUT}: train {len(tr):,} pairs, positives {int(tr['y'].sum()):,}", flush=True)
    # keep all positives + hardest negatives first (train pairs are the ~40/S1 blocking set)
    if len(tr) > max_train:                   # at most half positives, rest hard negatives (positives can exceed cap)
        pos, neg = tr.filter(pl.col("y") == 1), tr.filter(pl.col("y") == 0)
        n_pos = min(len(pos), max_train // 2)
        n_neg = min(len(neg), max_train - n_pos)
        tr = pl.concat([pos.sample(n_pos, seed=0), neg.sample(n_neg, seed=0)])
    tr = tr.sample(fraction=1.0, shuffle=True, seed=0)
    va = pl.read_parquet(CE / os.environ.get("CE_VAL_FILE", "val.parquet"))
    va = va.sample(n=200_000, seed=0) if len(va) > 200_000 else va
    dl = torch.utils.data.DataLoader(_dataset(tr, tok), batch_size=bs, shuffle=True, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = int(len(dl) * epochs)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, step = time.time(), 0
    for ep in range(int(np.ceil(epochs))):
        for ids, mask, y in dl:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits.squeeze(-1)
                loss = lossf(out.float(), y.cuda())
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            step += 1
            if step % 500 == 0:
                print(f"step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
            if step >= steps:
                break
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    p = _predict(model, tok, va)
    from sklearn.metrics import log_loss, roc_auc_score
    print(f"val: auc {roc_auc_score(va['y'], p):.5f} logloss {log_loss(va['y'], np.clip(p, 1e-6, 1 - 1e-6)):.5f}")


def _predict(model, tok, df, bs: int = 512):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(df), bs * 20):
            part = df.slice(i, bs * 20)
            enc = tok(part["text_a"].to_list(), part["text_b"].to_list(), truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt")
            for j in range(0, len(part), bs):
                with torch.autocast("cuda", dtype=torch.float16):
                    lo = model(input_ids=enc["input_ids"][j:j + bs].cuda(),
                               attention_mask=enc["attention_mask"][j:j + bs].cuda()).logits.squeeze(-1)
                out.append(torch.sigmoid(lo.float()).cpu().numpy())
    return np.concatenate(out)


def export_pairs(pairs_path: str, split: str, name: str):
    """Attach raw texts to an (s1, cand) pair list -> CE/<name>.parquet (input of `score`)."""
    CE.mkdir(parents=True, exist_ok=True)
    pairs = pl.read_parquet(pairs_path)
    pairs = pairs.select([c for c in ("s1", "cand", "y") if c in pairs.columns])
    raw = lambda s: pl.scan_parquet(WORK / "raw" / f"{split}_source{s}.parquet")
    txt = lambda df, col: df.select(pl.col("entity_id").alias(col), pl.concat_str(
        ["business_name", "business_address"], separator=" | ").alias("text_" + ("a" if col == "s1" else "b")))
    s1 = raw(1).filter(pl.col("entity_id").is_in(pairs["s1"].unique().implode())).collect()
    pool = pl.concat([raw(s).filter(pl.col("entity_id").is_in(pairs["cand"].unique().implode())).collect() for s in (2, 3)])
    d = pairs.join(txt(s1, "s1"), on="s1", how="left").join(txt(pool, "cand"), on="cand", how="left")
    d.write_parquet(CE / f"{name}.parquet")
    print(f"exported {name}: {len(d):,} pairs", flush=True)


def score(name: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(OUT)
    model = AutoModelForSequenceClassification.from_pretrained(OUT).cuda()
    df = pl.read_parquet(CE / f"{name}.parquet")
    k, n = map(int, os.environ.get("CE_SHARD", "0/1").split("/"))      # n GPU jobs, each scores 1/n
    step = -(-len(df) // n)
    df = df.slice(k * step, step)
    t0 = time.time()
    p = _predict(model, tok, df)
    df.select("s1", "cand").with_columns(pl.Series("ce", p.astype(np.float32)))       .write_parquet(OUT / f"{name}_scores_{k}.parquet")
    print(f"scored {name}: {len(df):,} pairs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "export":
        export()
    elif cmd == "train":
        train()
    elif cmd == "score":
        score(sys.argv[2])
    elif cmd == "split_folds":        # <name>: train pairs -> <name>_f0 / _f1 by S1 state (same folds as stage-2 OOF)
        from stage2 import state_folds
        d = pl.read_parquet(CE / f"{sys.argv[2]}.parquet")
        f = state_folds(d["s1"])
        for k in (0, 1):
            d.filter(pl.Series(f == k)).write_parquet(CE / f"{sys.argv[2]}_f{k}.parquet")
            print(f"fold {k}: {(f == k).sum():,} pairs", flush=True)
    elif cmd == "export_pairs":
        export_pairs(sys.argv[2], sys.argv[3], sys.argv[4])
