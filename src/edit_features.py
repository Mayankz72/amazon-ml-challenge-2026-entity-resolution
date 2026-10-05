"""Edit-evidence features: WHICH words / numbers differ between an S1 record and a candidate, and how likely
that kind of difference is under "same business with noise" vs "look-alike twin".

True noise: typos, abbreviations, shuffles, added generic words (center, holdings), decoy prefix numbers
(Door No 878). Twins: a real word swapped (UP Retails -> UP Export), house number changed (5591 -> 5604).

Word log-likelihood ratios are learned from TRAINING candidate pairs (labels y) — out-of-fold on train.
"""
import math
from collections import Counter

import numpy as np
import polars as pl
from rapidfuzz.distance import JaroWinkler, Levenshtein

JW_MATCH = 0.88


def align(a_toks, b_toks):
    """Greedy fuzzy alignment; returns (missing tokens of a, extra tokens of b, n matched, n typo matches)."""
    b_left = list(b_toks)
    missing, n_exact, n_typo = [], 0, 0
    for t in a_toks:
        if t in b_left:
            b_left.remove(t); n_exact += 1
            continue
        best, bj = None, 0.0
        for u in b_left:
            j = JaroWinkler.similarity(t, u)
            if j > bj:
                best, bj = u, j
        if best is not None and bj >= JW_MATCH:
            b_left.remove(best); n_typo += 1
        else:
            missing.append(t)
    return missing, b_left, n_exact, n_typo


def num_relation(a: str, b: str) -> int:
    """0 exact, 1 zero-pad, 2 prefix/suffix (truncation), 3 one edit, 4 unrelated."""
    if a == b:
        return 0
    if a.lstrip("0") == b.lstrip("0"):
        return 1
    if a.startswith(b) or b.startswith(a) or a.endswith(b) or b.endswith(a):
        return 2
    if Levenshtein.distance(a, b) <= 1:
        return 3
    return 4


def _pair_edits(n1, n2, a1, a2, num1, num2, h1, h2):
    ta, tb = n1.split(), n2.split()
    miss, extra, n_ex, n_ty = align(ta, tb)
    aa, ab = a1.split(), a2.split()
    amiss, aextra, _, _ = align([t for t in aa if not t.isdigit()], [t for t in ab if not t.isdigit()])
    na, nb = num1.split(), num2.split()
    # every S1 number: best relation to any candidate number; every candidate number: best relation to S1
    rel_a = [min((num_relation(x, y) for y in nb), default=5) for x in na]
    rel_b = [min((num_relation(y, x) for x in na), default=5) for y in nb]
    h_rel = num_relation(h1, h2) if h1 and h2 else 5
    return miss, extra, n_ex, n_ty, amiss, aextra, rel_a, rel_b, h_rel


def raw_edits(L: pl.DataFrame, R: pl.DataFrame):
    """Per pair: token-level edits (lists) + numeric summary features."""
    cols = zip(L["name_core"].to_list(), R["name_core"].to_list(), L["addr_clean"].to_list(),
               R["addr_clean"].to_list(), L["addr_nums"].to_list(), R["addr_nums"].to_list(),
               L["house"].to_list(), R["house"].to_list())
    miss_l, extra_l, amiss_l, aextra_l = [], [], [], []
    num = {k: [] for k in ("e_n_exact", "e_n_typo", "e_n_miss", "e_n_extra", "e_a_miss", "e_a_extra",
                           "e_num_a_unrel", "e_num_b_unrel", "e_num_a_best", "e_num_b_cnt", "e_house_rel")}
    for n1, n2, a1, a2, x1, x2, h1, h2 in cols:
        miss, extra, n_ex, n_ty, amiss, aextra, rel_a, rel_b, h_rel = _pair_edits(
            n1 or "", n2 or "", a1 or "", a2 or "", x1 or "", x2 or "", h1 or "", h2 or "")
        # stored as one space-joined string per pair (lists of lists cost GBs at 8M pairs)
        miss_l.append(" ".join(miss)); extra_l.append(" ".join(extra))
        amiss_l.append(" ".join(amiss)); aextra_l.append(" ".join(aextra))
        num["e_n_exact"].append(n_ex); num["e_n_typo"].append(n_ty)
        num["e_n_miss"].append(len(miss)); num["e_n_extra"].append(len(extra))
        num["e_a_miss"].append(len(amiss)); num["e_a_extra"].append(len(aextra))
        num["e_num_a_unrel"].append(sum(r == 4 for r in rel_a))
        num["e_num_b_unrel"].append(sum(r == 4 for r in rel_b))
        num["e_num_a_best"].append(min(rel_a) if rel_a else 5)
        num["e_num_b_cnt"].append(len(rel_b))
        num["e_house_rel"].append(h_rel)
    return {"miss": miss_l, "extra": extra_l, "amiss": amiss_l, "aextra": aextra_l}, num


def learn_llr(edits: dict, y: np.ndarray, min_count: int = 3) -> dict:
    """Word -> log P(word edited | true pair) / P(word edited | false pair), per edit kind."""
    tables = {}
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    for kind, lists in edits.items():
        cp, cn = Counter(), Counter()
        for toks, lab in zip(lists, y):
            if toks:
                (cp if lab else cn).update(set(toks.split()))
        tab = {}
        for t in set(cp) | set(cn):
            if cp[t] + cn[t] >= min_count:
                tab[t] = math.log((cp[t] + 0.5) / (pos + 1)) - math.log((cn[t] + 0.5) / (neg + 1))
        tables[kind] = tab
    return tables


def llr_features(edits: dict, tables: dict) -> dict:
    out = {}
    for kind, lists in edits.items():
        tab = tables.get(kind, {})
        s, mn, unk = [], [], []
        for joined in lists:
            toks = joined.split() if joined else []
            v = [tab[t] for t in toks if t in tab]
            s.append(sum(v) if v else 0.0)
            mn.append(min(v) if v else 0.0)
            unk.append(sum(1 for t in toks if t not in tab))
        out[f"e_llr_{kind}_sum"] = np.array(s, np.float32)
        out[f"e_llr_{kind}_min"] = np.array(mn, np.float32)
        out[f"e_unk_{kind}"] = np.array(unk, np.int16)
    return out


EDIT_STR = ["name_core", "addr_clean", "addr_nums", "house"]
