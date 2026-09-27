"""Pairwise features for (Source-1 record, Source-2/3 candidate) pairs.

MEMORY NOTE: the previous version built one Python list of 26-float tuples
for ALL pairs before handing it to pd.DataFrame(). At hackathon scale (can
be 100M+ candidate pairs after blocking) that list alone is tens of GB of
Python object overhead, on top of the DataFrame copy made from it. This
version writes straight into one preallocated float32 numpy array (no
intermediate list of tuples, no second copy), and processes it in chunks
so a partially-completed run's working set stays bounded. Chunks are
farmed out to a thread pool because rapidfuzz's C implementations release
the GIL, so this also uses spare cores instead of adding memory.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

NAN = float("nan")
DEFAULT_CHUNK = 200_000


def _jacc(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return NAN
    return len(sa & sb) / len(sa | sb)


def _acronym(a, b):
    ta, tb = a.split(), b.split()
    ca, cb = a.replace(" ", ""), b.replace(" ", "")
    ia, ib = "".join(t[0] for t in ta), "".join(t[0] for t in tb)
    return float((len(ta) > 1 and ia == cb) or (len(tb) > 1 and ib == ca))


def _pair_cos(M, i, j, chunk=DEFAULT_CHUNK):
    out = np.empty(len(i), dtype=np.float32)
    for s in range(0, len(i), chunk):
        prod = M[i[s:s + chunk]].multiply(M[j[s:s + chunk]])
        out[s:s + chunk] = np.asarray(prod.sum(axis=1)).ravel()
    return out


COLS = [
    "name_ratio", "name_tsort", "name_tset", "name_partial", "name_wratio", "name_jw", "name_lev",
    "name_jacc", "name_first_eq", "name_compact_eq", "name_acro", "name_len_diff", "name_ntok_diff",
    "name_digit_conflict", "legal_jacc", "name_full_tset",
    "addr_tset", "addr_tsort", "addr_partial", "addr_jacc", "addr_full_tset", "addr_missing",
    "postal_eq", "postal_p3", "nums_jacc", "country_match",
]
_NCOLS = len(COLS)


def _fill_chunk(out, s, e, i, j, nc, nl, nf, ac, af, pc, nu, cy):
    """Fill out[s:e, :] in place for pair positions [s, e). No allocation of
    per-pair Python objects beyond what rapidfuzz itself needs."""
    for row, k in enumerate(range(s, e)):
        a, b = i[k], j[k]
        n1, n2 = nc[a], nc[b]
        t1, t2 = n1.split(), n2.split()
        d1 = {t for t in t1 if t.isdigit()}
        d2 = {t for t in t2 if t.isdigit()}
        a1, a2 = ac[a], ac[b]
        addr_ok = bool(a1 and a2)
        p1, p2 = pc[a], pc[b]
        p_ok = bool(p1 and p2)
        l1, l2 = nl[a], nl[b]
        out[s + row, 0] = fuzz.ratio(n1, n2) / 100
        out[s + row, 1] = fuzz.token_sort_ratio(n1, n2) / 100
        out[s + row, 2] = fuzz.token_set_ratio(n1, n2) / 100
        out[s + row, 3] = fuzz.partial_ratio(n1, n2) / 100
        out[s + row, 4] = fuzz.WRatio(n1, n2) / 100
        out[s + row, 5] = JaroWinkler.similarity(n1, n2)
        out[s + row, 6] = Levenshtein.normalized_similarity(n1, n2)
        out[s + row, 7] = _jacc(n1, n2)
        out[s + row, 8] = float(bool(t1 and t2 and t1[0] == t2[0]))
        out[s + row, 9] = float(n1.replace(" ", "") == n2.replace(" ", ""))
        out[s + row, 10] = _acronym(n1, n2)
        out[s + row, 11] = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)
        out[s + row, 12] = abs(len(t1) - len(t2))
        out[s + row, 13] = float(bool(d1 and d2 and d1 != d2))
        out[s + row, 14] = _jacc(l1, l2) if (l1 and l2) else -1.0
        out[s + row, 15] = fuzz.token_set_ratio(nf[a], nf[b]) / 100
        out[s + row, 16] = fuzz.token_set_ratio(a1, a2) / 100 if addr_ok else NAN
        out[s + row, 17] = fuzz.token_sort_ratio(a1, a2) / 100 if addr_ok else NAN
        out[s + row, 18] = fuzz.partial_ratio(a1, a2) / 100 if addr_ok else NAN
        out[s + row, 19] = _jacc(a1, a2) if addr_ok else NAN
        out[s + row, 20] = fuzz.token_set_ratio(af[a], af[b]) / 100 if (af[a] and af[b]) else NAN
        out[s + row, 21] = float(not addr_ok)
        out[s + row, 22] = float(p1 == p2) if p_ok else NAN
        out[s + row, 23] = float(p1[:3] == p2[:3]) if p_ok else NAN
        out[s + row, 24] = _jacc(nu[a], nu[b]) if (nu[a] and nu[b]) else NAN
        out[s + row, 25] = float(cy[a] == cy[b])


def build_features(rec, pairs, mats, chunk_size=DEFAULT_CHUNK, n_workers=None):
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    n = len(i)
    nc, nl, nf = rec["name_core"].to_numpy(), rec["name_legal"].to_numpy(), rec["name_full"].to_numpy()
    ac, af = rec["addr_core"].to_numpy(), rec["addr_full"].to_numpy()
    pc, nu, cy = rec["postal"].to_numpy(), rec["nums"].to_numpy(), rec["ctry"].to_numpy()

    arr = np.empty((n, _NCOLS), dtype=np.float32)
    bounds = list(range(0, n, chunk_size)) + [n]
    n_workers = n_workers or min(32, (os.cpu_count() or 4))
    if n <= chunk_size or n_workers <= 1:
        for s, e in zip(bounds[:-1], bounds[1:]):
            _fill_chunk(arr, s, e, i, j, nc, nl, nf, ac, af, pc, nu, cy)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_fill_chunk, arr, s, e, i, j, nc, nl, nf, ac, af, pc, nu, cy)
                    for s, e in zip(bounds[:-1], bounds[1:])]
            for f in futs:
                f.result()

    F = pd.DataFrame(arr, columns=COLS)
    del arr
    F["cos_name_char"] = _pair_cos(mats["name_char"], i, j, chunk_size)
    F["cos_name_word"] = _pair_cos(mats["name_word"], i, j, chunk_size)
    F["cos_full_char"] = _pair_cos(mats["full_char"], i, j, chunk_size)
    F["cos_addr_char"] = _pair_cos(mats["addr_char"], i, j, chunk_size)
    return F


def _competition(df, key, prefix):
    """How does this pair compare with the other candidates of the same group?"""
    grp = [df[k] for k in key]
    g = df.groupby(key)["sc"]
    top1 = g.transform("max")
    tie = (df["sc"] == top1).groupby(grp).transform("sum") > 1
    below = df["sc"].where(df["sc"] < top1)
    second = below.groupby(grp).transform("max").fillna(0.0)
    second = np.where(tie, top1, second)
    df[f"{prefix}_gap_top"] = top1 - df["sc"]
    df[f"{prefix}_margin"] = df["sc"] - np.where(df["sc"] >= top1, second, top1)
    df[f"{prefix}_rank"] = g.rank(ascending=False, method="min")
    df[f"{prefix}_n"] = g.transform("size")
    return df


def add_competition(df):
    """Adds a scalar pair score plus S1-side / other-side competition features.
    Source 1 is deduplicated, so a S2/S3 record usually belongs to ONE S1 entity:
    a pair that loses to a rival S1 entity is unlikely to be a true match."""
    df["sc"] = (df["cos_full_char"] + df["cos_name_char"] + df["name_wratio"]) / 3
    df = _competition(df, ["i", "src"], "s1")
    df = _competition(df, ["j"], "rev")
    return df