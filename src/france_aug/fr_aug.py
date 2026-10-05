"""Synthetic FRENCH training pairs with TRUE labels: rewrite labeled US CE pairs into French surface forms.
Both sides of a pair get the SAME rewrite (same city, region, street type, legal form...), so a match stays a match
and a decoy twin stays a twin. Vocabulary (French cities + their regions) comes from the test file's own France
addresses (no external data). Output: ce/fraug_train.parquet (mix with original pairs), ce/fraug_val.parquet.
"""
import os as _os
AMZ = _os.environ.get("AMZML_HOME", _os.path.expanduser("~/amzml"))
TMP = _os.environ.get("AMZML_TMP", "/tmp/amzml")

import random
import re
import zlib

import sys
from collections import Counter

import polars as pl

sys.path.insert(0, AMZ + "/code/src")
from text import US_STATES          # full state name -> code (our normaliser's table)
CODE2NAMES = {}
for n, c in US_STATES.items():
    CODE2NAMES.setdefault(c.upper(), []).append(n)

A = AMZ + "/work/"
REGIONS = {"hauts-de-france": ["Nord", "Pas-de-Calais"], "nouvelle-aquitaine": ["Gironde", "Landes"],
           "pays de la loire": ["Loire-Atlantique", "Vendée"], "ile-de-france": ["Paris"], "île-de-france": ["Paris"]}

# ---- France vocabulary from the test S1 addresses: "..., City, Region"
fr = pl.read_parquet(A + "raw/test_source1.parquet").filter(pl.col("country") == "France")["business_address"].to_list()
cities, freq = {}, Counter()
for a in fr:
    parts = [p.strip() for p in (a or "").split(",") if p.strip()]
    for i, p in enumerate(parts):
        if p.lower() in REGIONS and i > 0 and not re.search(r"\d", parts[i - 1]):
            cities[parts[i - 1]] = p
            freq[parts[i - 1]] += 1
CITY_LIST = sorted(c for c in cities if freq[c] >= 300 and 2 < len(c) < 30)
print(f"French cities from test: {len(CITY_LIST)}: {CITY_LIST}", flush=True)

STREET = {  # English street type -> French variants (first = full, others = abbreviations seen in France data)
    "street": ["Rue", "R.", "R"], "st": ["Rue", "R."], "road": ["Route", "Rte."], "rd": ["Route", "Rte"],
    "avenue": ["Avenue", "Av.", "AV"], "ave": ["Avenue", "Av."], "boulevard": ["Boulevard", "Bd", "BD"],
    "blvd": ["Boulevard", "Bd"], "lane": ["Allée", "All."], "ln": ["Allée", "All."], "drive": ["Chemin", "Ch."],
    "dr": ["Chemin", "Ch."], "court": ["Impasse", "Imp."], "ct": ["Impasse", "Imp."], "place": ["Place", "Pl."],
    "pl": ["Place", "Pl."], "way": ["Cours", "Crs"], "circle": ["Square", "Sq."], "parkway": ["Quai", "Qu."],
    "highway": ["Route Nationale", "RN"], "trail": ["Sentier", "Sent."], "terrace": ["Cité", "Cite"]}
WORDS = {  # names: legal forms + generic business words (whole words)
    "llc": "SARL", "l.l.c.": "SARL", "inc": "SAS", "inc.": "SAS", "incorporated": "SAS", "corp": "SA",
    "corp.": "SA", "corporation": "SA", "ltd": "EURL", "ltd.": "EURL", "limited": "EURL", "llp": "SCI",
    "pllc": "SELARL", "co": "Cie", "co.": "Cie", "company": "Compagnie", "and": "et", "sons": "Fils",
    "brothers": "Frères", "associates": "Associés", "school": "École", "clinic": "Clinique",
    "pharmacy": "Pharmacie", "hospital": "Hôpital", "sports": "Sportive", "friends": "Amis", "society": "Société",
    "enterprises": "Entreprises", "house": "Maison", "consulting": "Conseil", "bakery": "Boulangerie",
    "hotel": "Hôtel", "foundation": "Fondation", "group": "Groupe", "center": "Centre", "centre": "Centre",
    "health": "Santé", "services": "Services", "solutions": "Solutions", "church": "Église", "club": "Club",
    "restaurant": "Restaurant", "cafe": "Café", "shop": "Boutique", "store": "Magasin", "studio": "Atelier",
    "workshop": "Atelier", "academy": "Académie", "college": "Collège", "committee": "Comité",
    "association": "Association", "partners": "Partenaires", "management": "Gestion", "development": "Développement",
    "home": "Maison", "care": "Soins", "dental": "Dentaire", "medical": "Médical", "law": "Avocats",
    "financial": "Financement", "insurance": "Assurances", "auto": "Auto", "garage": "Garage", "tech": "Tech",
    "global": "Global", "international": "International", "the": "Le"}
US_STATE = re.compile(r"\b[A-Z]{2}\b")
TYPE_RE = re.compile(r"\b(\d+[A-Za-z]?)\s+((?:[A-Za-z0-9'\.]+\s+){0,3}?[A-Za-z0-9'\.]+?)\s+(" +
                     "|".join(sorted(STREET, key=len, reverse=True)) + r")\.?\b", re.I)
WORD_RE = re.compile(r"(?<![\w.])(" + "|".join(re.escape(w) for w in sorted(WORDS, key=len, reverse=True)) + r")(?![\w])", re.I)


def case_like(src: str, dst: str) -> str:
    return dst.upper() if src.isupper() and len(src) > 1 else dst


def make_rewrite(key: str, text_a: str):
    """Pair-specific but deterministic choices; returns f(text) applied identically to both sides."""
    rng = random.Random(zlib.crc32(key.encode()))
    variant = rng.random()
    # US city + state of the S1 side -> one French city + its region (sometimes the departement, as in the data)
    addr = text_a.split("|", 1)[1] if "|" in text_a else ""
    comps = [c.strip() for c in addr.split(",") if c.strip()]
    city = state = None
    for i, c in enumerate(comps):
        if US_STATE.fullmatch(c) and i > 0 and not re.search(r"\d", comps[i - 1]):
            city, state = comps[i - 1], c
    fcity = rng.choice(CITY_LIST)
    freg = cities[fcity]
    freg_alt = rng.choice(REGIONS.get(freg.lower(), [freg])) if variant < 0.3 else freg

    def f(t: str) -> str:
        if city:
            t = re.sub(r"\b" + re.escape(city) + r"\b", lambda m: case_like(m.group(0), fcity), t, flags=re.I)
        if state:                            # same replacement on both sides of the pair
            t = re.sub(r"\b" + state + r"\b", freg_alt, t)
            for nm in CODE2NAMES.get(state, []):
                t = re.sub(r"\b" + re.escape(nm) + r"\b", lambda m: case_like(m.group(0), freg_alt), t, flags=re.I)
        t = TYPE_RE.sub(lambda m: f"{m.group(1)} {case_like(m.group(3), STREET[m.group(3).lower()][0 if variant < 0.5 else -1])} "
                                  f"{m.group(2)}", t)
        t = WORD_RE.sub(lambda m: case_like(m.group(1), WORDS[m.group(1).lower()]), t)
        return t
    return f


def augment(df: pl.DataFrame) -> pl.DataFrame:
    out_a, out_b = [], []
    for s1, cand, a, b in df.select("s1", "cand", "text_a", "text_b").iter_rows():
        f = make_rewrite(s1, a)              # keyed by S1: every pair of an S1 gets the same French "place"
        out_a.append(f(a)); out_b.append(f(b or ""))
    return df.with_columns(pl.Series("text_a", out_a), pl.Series("text_b", out_b))


if __name__ == "__main__":
    tr = pl.read_parquet(A + "ce/rest_train_US.parquet")
    pos, neg = tr.filter(pl.col("y") == 1), tr.filter(pl.col("y") == 0)
    fr_tr = augment(pl.concat([pos.sample(250_000, seed=7), neg.sample(250_000, seed=7)]))
    for r in fr_tr.sample(12, seed=3).iter_rows(named=True):
        print(r["y"], "|", r["text_a"], "  <->  ", r["text_b"])
    orig = pl.read_parquet(A + "ce/rest_train.parquet")
    orig = pl.concat([orig.filter(pl.col("y") == 1).sample(150_000, seed=8), orig.filter(pl.col("y") == 0).sample(150_000, seed=8)])
    mix = pl.concat([fr_tr.select(orig.columns), orig]).with_columns(pl.col("y").cast(pl.Int8)).sample(fraction=1.0, shuffle=True, seed=9)
    mix.write_parquet(A + "ce/fraug_train.parquet")
    va = augment(pl.read_parquet(A + "ce/rest_val_US.parquet"))
    va.write_parquet(A + "ce/fraug_val.parquet")
    print(f"FRAUG train {len(mix):,} (500k French-rewritten US + 300k original), val {len(va):,}", flush=True)
