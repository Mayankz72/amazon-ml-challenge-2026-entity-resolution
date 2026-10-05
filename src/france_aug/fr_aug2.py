"""Synthetic French pairs, pass 2: fr_aug rewrite (same on both sides) + REALISTIC France record noise on the S2/S3 side
only (labels stay true): house-number formats (N°, #, 00-padding, bis/ter), region <-> departement, dropped accents,
UPPERCASE, street abbreviations, and '& Fils' / 'Et Fils' added to TRUE matches (French Fils pairs are mostly
genuine). 700k French pairs + 300k original. Output ce/fraug2_train.parquet, ce/fraug2_val.parquet."""
import os as _os
AMZ = _os.environ.get("AMZML_HOME", _os.path.expanduser("~/amzml"))
TMP = _os.environ.get("AMZML_TMP", "/tmp/amzml")

import random
import re
import sys
import unicodedata
import zlib

import polars as pl

sys.path.insert(0, AMZ + "/v6r")
import fr_aug as fa

DEPT = {"Hauts-de-France": ["Nord", "Pas-de-Calais"], "Nouvelle-Aquitaine": ["Gironde", "Landes"],
        "Pays de la Loire": ["Loire-Atlantique", "Vendée"]}
ABBR = {"Rue": ["R.", "R", "RUE"], "Avenue": ["Av.", "AV", "Ave"], "Boulevard": ["Bd", "BD", "Bld"],
        "Allée": ["All.", "Allee"], "Route": ["Rte", "Rte."], "Impasse": ["Imp.", "Imp"], "Place": ["Pl.", "PL"],
        "Chemin": ["Ch.", "Chem."], "Cours": ["Crs", "Crs."]}
LEGAL = ["SARL", "SAS", "SA", "EURL", "SCI", "SASU", "EI", "SNC"]


def strip_acc(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def noise(t: str, y: int, rng: random.Random) -> str:
    if "|" not in t:
        return t
    name, addr = [x.strip() for x in t.split("|", 1)]
    r = rng.random
    if addr:
        if r() < 0.25:
            addr = re.sub(r"\b(\d+)\b", lambda m: rng.choice([f"N°{m.group(1)}", f"#{m.group(1)}", m.group(1).zfill(4),
                                                             f"No {m.group(1)}"]), addr, count=1)
        if r() < 0.08:
            addr = re.sub(r"\b(\d+)\b", lambda m: m.group(1) + rng.choice([" bis", " ter", "B"]), addr, count=1)
        for reg, deps in DEPT.items():
            if reg in addr and r() < 0.4:
                addr = addr.replace(reg, rng.choice(deps))
        for full, ab in ABBR.items():
            if full in addr and r() < 0.4:
                addr = addr.replace(full, rng.choice(ab), 1)
        if r() < 0.3:
            addr = strip_acc(addr)
        if r() < 0.25:
            addr = addr.upper()
        if r() < 0.2:                                   # component reordering (seen in France data)
            parts = [p.strip() for p in addr.split(",")]
            rng.shuffle(parts)
            addr = ", ".join(parts)
    if y == 1 and r() < 0.12:                           # French generator: family suffix on genuine records
        name = name + rng.choice([" & Fils", " Et Fils", " & Associés", " & Frères", " SARL & Fils"])
    if r() < 0.15:
        toks = name.split()
        legal = [w for w in toks if w.upper().strip(".") in LEGAL]
        if legal:
            name = " ".join(w for w in toks if w not in legal) + " " + rng.choice(LEGAL)
    if r() < 0.2:
        name = strip_acc(name)
    if r() < 0.2:
        name = name.upper()
    return f"{name} | {addr}"


def augment2(df: pl.DataFrame) -> pl.DataFrame:
    base = fa.augment(df)
    out = []
    for s1, cand, y, b in base.select("s1", "cand", "y", "text_b").iter_rows():
        rng = random.Random(zlib.crc32((s1 + cand).encode()))
        out.append(noise(b or "", int(y), rng))
    return base.with_columns(pl.Series("text_b", out))


if __name__ == "__main__":
    A = fa.A
    tr = pl.read_parquet(A + "ce/rest_train_US.parquet")
    pos, neg = tr.filter(pl.col("y") == 1), tr.filter(pl.col("y") == 0)
    fr_tr = augment2(pl.concat([pos.sample(350_000, seed=17), neg.sample(350_000, seed=17)]))
    for r in fr_tr.sample(10, seed=4).iter_rows(named=True):
        print(r["y"], "|", r["text_a"], "  <->  ", r["text_b"])
    orig = pl.read_parquet(A + "ce/rest_train.parquet")
    orig = pl.concat([orig.filter(pl.col("y") == 1).sample(150_000, seed=18), orig.filter(pl.col("y") == 0).sample(150_000, seed=18)])
    mix = pl.concat([fr_tr.select(orig.columns), orig]).with_columns(pl.col("y").cast(pl.Int8)).sample(fraction=1.0, shuffle=True, seed=19)
    mix.write_parquet(A + "ce/fraug2_train.parquet")
    va = augment2(pl.read_parquet(A + "ce/rest_val_US.parquet"))
    va.write_parquet(A + "ce/fraug2_val.parquet")
    print(f"FRAUG2 train {len(mix):,}, val {len(va):,}", flush=True)
