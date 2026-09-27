"""Multi-strategy blocking / candidate generation.

Blocking combines several INDEPENDENT signals so that a match only has to be
caught by ONE of them, not all of them:

  1. TF-IDF nearest-neighbour blocking (char/word n-grams).
  2. Inverted-index TOKEN blocking on the business name (document-frequency
     capped so common words don't blow up candidate lists).
  3. PHONETIC blocking (NYSIIS code of the compacted name).
  4. EXACT-KEY blocking on postal code, and on (house/street number + first
     address word).
  5. A COUNTRY-RELAXED FALLBACK PASS for Source-1 records left with zero
     candidates after 1-4 within their own country.

MEMORY DESIGN - this is the part that matters for large real datasets:

  A given (i, j) candidate pair can only ever be produced by exactly ONE
  (source, country) iteration of the main loop below: Source-2 and Source-3
  row ranges never overlap, and every Source-1 row belongs to exactly one
  country group. That means we do NOT need to accumulate raw hits into one
  giant global structure (a dict of (i, j) -> set(), as in earlier
  versions) before deduplicating at the very end. Instead, each iteration's
  raw hits (which can have a lot of duplication across the several TF-IDF
  views + index lookups) are aggregated down to distinct (i, j) pairs
  IMMEDIATELY, right after that iteration finishes - so peak memory is
  bounded by one iteration's worth of raw hits, not the whole run's.

  Concretely: no Python dict/set is used as the accumulator at all. Every
  pass appends flat (i, j, strategy, score) arrays to a small DataFrame,
  which is aggregated with a single vectorized pandas groupby before moving
  to the next iteration. Only the small, already-deduplicated per-iteration
  results are kept around; everything else is released (falls out of scope)
  as soon as the iteration finishes.
"""
import gc
from collections import defaultdict

import jellyfish
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

REPS = {
    "name_char": ("name_core", dict(analyzer="char_wb", ngram_range=(2, 4), n_features=2 ** 15)),
    "name_word": ("name_core", dict(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", n_features=2 ** 14)),
    "full_char": ("text_full", dict(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 16)),
    "addr_char": ("addr_core", dict(analyzer="char_wb", ngram_range=(3, 4), n_features=2 ** 15)),
}
DEFAULT_K = {"name_char": 12, "name_word": 6, "full_char": 12}
DEFAULT_CAP_PER_ENTITY = 100
TOKEN_MIN_LEN = 3
# TOKEN_MAX_DF_RATIO alone is not enough at multi-million-row scale: 3% of a
# 2M-record target group is still 60,000 rows for one common token, and if
# 100k S1 rows share that token you get 6 BILLION raw hit rows before any
# aggregation. TOKEN_MAX_DF_ABS puts a hard ceiling on that regardless of
# group size, and MAX_HITS_PER_TOKEN additionally caps how many of a given
# token's postings any single row is allowed to fan out against.
TOKEN_MAX_DF_RATIO = 0.003
TOKEN_MAX_DF_ABS = 2_000
TOKEN_MIN_DF_CAP = 3
MAX_HITS_PER_TOKEN = 50
TOPK_MAX_DENSE_CHUNK = 2_000_000  # cap on dense-array elements materialised at once in topk_rows


def build_matrices(rec, chunk_size=2000):
    """HashingVectorizer avoids storing a vocabulary (unlike TfidfVectorizer),
    which is the right call for a large corpus - keep this design. Chunked
    transform bounds peak memory during matrix construction itself."""
    mats = {}
    for name, (col, kw) in REPS.items():
        options = dict(kw)
        n_features = options.pop("n_features")
        vec = HashingVectorizer(dtype=np.float32, alternate_sign=False, n_features=n_features, **options)
        tf = TfidfTransformer(sublinear_tf=True, use_idf=False)
        chunks = []
        for start in range(0, len(rec), chunk_size):
            counts = vec.transform(rec[col].iloc[start:start + chunk_size])
            chunks.append(tf.fit_transform(counts).tocsr())
        mats[name] = vstack(chunks, format="csr")
        del chunks
    return mats


def topk_rows(A, B, k):
    """Top-k columns of B for every row of A by dot product (chunked so the
    temporary dense array never exceeds TOPK_MAX_DENSE_CHUNK elements)."""
    n, m = A.shape[0], B.shape[0]
    k = min(k, m)
    idx = np.empty((n, k), dtype=np.int64)
    sim = np.empty((n, k), dtype=np.float32)
    Bt = B.T.tocsr()
    chunk = max(1, TOPK_MAX_DENSE_CHUNK // max(m, 1))
    for s in range(0, n, chunk):
        S = (A[s:s + chunk] @ Bt).toarray()
        if k < m:
            part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        else:
            part = np.tile(np.arange(m), (S.shape[0], 1))
        idx[s:s + chunk] = part
        sim[s:s + chunk] = np.take_along_axis(S, part, axis=1)
        del S
    return idx, sim


def _tfidf_chunk(mats, s, t, rep, k, tag):
    idx, sim = topk_rows(mats[rep][s], mats[rep][t], k)
    ii = np.repeat(s, idx.shape[1])
    jj = t[idx.ravel()]
    ss = sim.ravel()
    keep = ss > 0
    return pd.DataFrame({"i": ii[keep], "j": jj[keep], "strategy": tag, "score": ss[keep]})


# ---------------------------------------------------------------- index-based blocking

def _tokenize(value, minlen=TOKEN_MIN_LEN):
    return [t for t in value.split() if len(t) >= minlen]


def _phonetic(value):
    compact = value.replace(" ", "")
    if len(compact) < 3:
        return ""
    try:
        return jellyfish.nysiis(compact)
    except Exception:
        return ""


def _addr_key(addr_core, nums):
    toks = addr_core.split()
    first_word = toks[0] if toks else ""
    first_num = nums.split()[0] if nums else ""
    return f"{first_num}_{first_word}" if (first_num and first_word) else ""


def build_indexes(rec, mask, max_df_ratio=TOKEN_MAX_DF_RATIO, min_df_cap=TOKEN_MIN_DF_CAP):
    pos = np.where(mask)[0]
    token_idx, phon_idx, postal_idx, addr_idx = (defaultdict(list) for _ in range(4))
    name_core = rec["name_core"].to_numpy()
    addr_core = rec["addr_core"].to_numpy()
    postal = rec["postal"].to_numpy()
    nums = rec["nums"].to_numpy()
    for p in pos:
        for t in set(_tokenize(name_core[p])):
            token_idx[t].append(p)
        ph = _phonetic(name_core[p])
        if ph:
            phon_idx[ph].append(p)
        if postal[p] and len(postal[p]) >= 4:
            postal_idx[postal[p]].append(p)
        ak = _addr_key(addr_core[p], nums[p])
        if ak:
            addr_idx[ak].append(p)
    if pos.size:
        cap = min(max(min_df_cap, int(max_df_ratio * pos.size)), TOKEN_MAX_DF_ABS)
        token_idx = {t: v for t, v in token_idx.items() if len(v) <= cap}
    return token_idx, phon_idx, postal_idx, addr_idx


def _index_pass_chunk(s_idx, name_core, addr_core, postal, nums,
                      token_idx, phon_idx, postal_idx, addr_idx, tag_prefix=""):
    """One index-based blocking pass. Returns a flat (i, j, strategy, score)
    DataFrame chunk - score is 0.0 for these (0/1 hit, no similarity value)."""
    I, J, TAG = [], [], []
    for i in s_idx:
        for t in set(_tokenize(name_core[i])):
            # even after the df cap in build_indexes, cap how many postings a
            # SINGLE row can fan out against - this is what actually bounds
            # worst-case output size per row, independent of corpus size.
            for c in token_idx.get(t, ())[:MAX_HITS_PER_TOKEN]:
                I.append(i); J.append(c); TAG.append(tag_prefix + "token")
        ph = _phonetic(name_core[i])
        if ph:
            for c in phon_idx.get(ph, ())[:MAX_HITS_PER_TOKEN]:
                I.append(i); J.append(c); TAG.append(tag_prefix + "phonetic")
        p = postal[i]
        if p and len(p) >= 4:
            for c in postal_idx.get(p, ())[:MAX_HITS_PER_TOKEN]:
                I.append(i); J.append(c); TAG.append(tag_prefix + "postal")
        ak = _addr_key(addr_core[i], nums[i])
        if ak:
            for c in addr_idx.get(ak, ())[:MAX_HITS_PER_TOKEN]:
                I.append(i); J.append(c); TAG.append(tag_prefix + "addr_key")
    if not I:
        return pd.DataFrame({"i": pd.Series(dtype=np.int64), "j": pd.Series(dtype=np.int64),
                             "strategy": pd.Series(dtype=object), "score": pd.Series(dtype=np.float32)})
    return pd.DataFrame({"i": np.asarray(I, dtype=np.int64), "j": np.asarray(J, dtype=np.int64),
                         "strategy": TAG, "score": np.float32(0.0)})


def _aggregate_iteration(chunks, cap_per_entity):
    """Collapse ONE iteration's raw hit chunks (which may contain a lot of
    duplication across views/strategies) down to distinct (i, j) pairs with
    n_strategies/block_score, and drop the raw chunks immediately. This is
    what keeps peak memory bounded to one iteration's worth of raw hits."""
    raw = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        {"i": pd.Series(dtype=np.int64), "j": pd.Series(dtype=np.int64),
         "strategy": pd.Series(dtype=object), "score": pd.Series(dtype=np.float32)})
    chunks.clear()
    if raw.empty:
        return pd.DataFrame({"i": pd.Series(dtype=np.int64), "j": pd.Series(dtype=np.int64),
                             "n_strategies": pd.Series(dtype=np.int64), "block_score": pd.Series(dtype=np.float32)})
    agg = raw.groupby(["i", "j"], sort=False).agg(
        n_strategies=("strategy", "nunique"), block_score=("score", "max")).reset_index()
    del raw
    if cap_per_entity:
        agg = (agg.sort_values(["i", "n_strategies", "block_score"], ascending=[True, False, False])
                  .groupby("i", sort=False).head(cap_per_entity).reset_index(drop=True))
    return agg


# ---------------------------------------------------------------- orchestration

def generate_candidates(rec, mats, ks=None, same_country=True, cap_per_entity=DEFAULT_CAP_PER_ENTITY,
                        s1_positions=None):
    """s1_positions: optional array of row positions (subset of the src==1
    rows) to process. Passing a batch here - instead of always the full S1
    set - is what lets pipeline.py process the dataset in bounded-memory
    chunks: the target side (S2/S3) is unchanged, only how much of S1 is
    matched against it in one call shrinks."""
    ks = ks or DEFAULT_K
    src, ctry = rec["src"].to_numpy(), rec["ctry"].to_numpy()
    name_core, addr_core = rec["name_core"].to_numpy(), rec["addr_core"].to_numpy()
    postal, nums = rec["postal"].to_numpy(), rec["nums"].to_numpy()
    n = len(rec)
    pos1 = np.where(src == 1)[0] if s1_positions is None else np.asarray(s1_positions)

    results = []  # small, already-deduplicated per-iteration DataFrames

    groups = np.unique(ctry[pos1]) if same_country else [None]
    for o in (2, 3):
        posO = np.where(src == o)[0]
        if posO.size == 0:
            continue
        for c in groups:
            s_mask = np.zeros(n, dtype=bool); s_mask[pos1] = True
            t_mask = np.zeros(n, dtype=bool); t_mask[posO] = True
            if c is not None:
                s_mask &= (ctry == c)
                t_mask &= (ctry == c)
            s, t = np.where(s_mask)[0], np.where(t_mask)[0]
            if s.size == 0 or t.size == 0:
                continue

            chunks = [_tfidf_chunk(mats, s, t, rep, k, f"tfidf:{rep}") for rep, k in ks.items()]
            token_idx, phon_idx, postal_idx, addr_idx = build_indexes(rec, t_mask)
            chunks.append(_index_pass_chunk(s, name_core, addr_core, postal, nums,
                                            token_idx, phon_idx, postal_idx, addr_idx))
            del token_idx, phon_idx, postal_idx, addr_idx

            # aggregate THIS iteration down to distinct pairs now, then drop the raw hits
            results.append(_aggregate_iteration(chunks, cap_per_entity))
            del chunks

    covered = set()
    for r in results:
        covered.update(r["i"].unique().tolist())
    leftover = np.array([i for i in pos1 if i not in covered])
    if leftover.size:
        for o in (2, 3):
            posO = np.where(src == o)[0]
            if posO.size == 0:
                continue
            full_mask = np.zeros(n, dtype=bool); full_mask[posO] = True
            token_idx, phon_idx, postal_idx, addr_idx = build_indexes(rec, full_mask)
            chunks = [_tfidf_chunk(mats, leftover, posO, rep, max(5, k // 2), f"fallback:tfidf:{rep}")
                     for rep, k in ks.items()]
            chunks.append(_index_pass_chunk(leftover, name_core, addr_core, postal, nums,
                                            token_idx, phon_idx, postal_idx, addr_idx, tag_prefix="fallback:"))
            del token_idx, phon_idx, postal_idx, addr_idx
            results.append(_aggregate_iteration(chunks, cap_per_entity))
            del chunks

    if not results or all(r.empty for r in results):
        return pd.DataFrame({"i": pd.Series(dtype=np.int64), "j": pd.Series(dtype=np.int64),
                             "n_strategies": pd.Series(dtype=np.int64), "block_score": pd.Series(dtype=np.float32)})

    # each (i, j) pair only ever comes from ONE iteration (Source-2/Source-3 row
    # ranges never overlap, and each S1 row has exactly one country group), so
    # this final concat needs no further cross-iteration merge - just stack.
    final = pd.concat(results, ignore_index=True)
    del results
    gc.collect()

    if cap_per_entity:
        final = (final.sort_values(["i", "n_strategies", "block_score"], ascending=[True, False, False])
                     .groupby("i", sort=False).head(cap_per_entity).reset_index(drop=True))
    return final