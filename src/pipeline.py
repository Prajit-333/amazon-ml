"""Shared helpers: loading, candidate+feature construction, decoding, metric."""
import csv
import gc

import numpy as np
import pandas as pd

from blocking import DEFAULT_CAP_PER_ENTITY, DEFAULT_K, build_matrices, generate_candidates
from features import add_competition, build_features
from normalize import prepare

NON_FEATURES = {"i", "j", "s1_id", "cand_id", "y", "proba"}


def read_tsv(path):
    # explicit tab separator; keep empty strings instead of NaN.
    #
    # Two deliberate choices here, each fixing a real, separate failure mode
    # on large/messy real-world TSVs:
    #   - dtype=object (NOT dtype=str): on some pandas versions/configs,
    #     dtype=str is routed through pandas' newer StringDtype extension-array
    #     construction, which is measurably slower and more memory-hungry than
    #     the classic numpy object-array path - this alone can turn a normal
    #     load into a multi-hour hang on a large file. dtype=object gives you
    #     the same plain Python strings without that overhead.
    #   - quoting=csv.QUOTE_NONE: business_name/business_address fields can
    #     contain stray " characters (inch marks, quoted nicknames, broken
    #     data). Without this, pandas applies CSV-style quoting semantics to
    #     ", and a field that opens a quote without a matching close can make
    #     the parser swallow large parts of the file into one runaway field.
    return pd.read_csv(
        path, sep="\t", dtype=object, keep_default_na=False,
        quoting=csv.QUOTE_NONE, engine="c",
    )


def load_records(split_dir, split):
    parts = []
    for s in (1, 2, 3):
        d = read_tsv(f"{split_dir}/{split}_source{s}.tsv")
        d["src"] = s
        parts.append(d)
    return prepare(pd.concat(parts, ignore_index=True))


def load_truth(path):
    gt = read_tsv(path)
    truth = {}
    for s1, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        truth[s1] = {x.strip() for x in ids.split(",") if x.strip()}
    return truth


def _assemble(rec, pairs, mats):
    """pairs+features -> one df with src/s1_id/cand_id attached, competition
    features computed. Shared by the single-shot and batched paths."""
    F = build_features(rec, pairs, mats)
    df = pd.concat([pairs, F], axis=1)
    del F
    df["src"] = rec["src"].to_numpy()[df["j"].to_numpy()]
    df = add_competition(df)
    ids = rec["entity_id"].to_numpy()
    df["s1_id"] = ids[df["i"].to_numpy()]
    df["cand_id"] = ids[df["j"].to_numpy()]
    return df


def candidates_and_features(rec, same_country=True, ks=None, cap_per_entity=DEFAULT_CAP_PER_ENTITY):
    """Single-shot, whole-dataset version. Fine at small/medium scale. At
    multi-million-row scale this holds ALL candidate pairs x ALL features in
    memory simultaneously and is the main source of OOM crashes - use
    iter_candidate_batches / build_training_table / predict_in_batches
    (below) instead once the dataset is large."""
    mats = build_matrices(rec)
    pairs = generate_candidates(rec, mats, ks or DEFAULT_K, same_country, cap_per_entity).reset_index(drop=True)
    df = _assemble(rec, pairs, mats)
    del mats, pairs  # TF-IDF matrices are not needed past this point - free them before training
    gc.collect()
    return df


DEFAULT_S1_BATCH = 20_000


def iter_s1_batches(rec, batch_size=DEFAULT_S1_BATCH):
    """Yield successive position arrays (row positions into `rec`) covering
    all src==1 rows, batch_size at a time. Batching S1 - not S2/S3 - is what
    bounds memory: each batch is matched against the FULL target set, so
    blocking recall is identical to the single-shot path; only how much of
    S1 is resident (candidates + features) at once shrinks."""
    pos1 = np.where(rec["src"].to_numpy() == 1)[0]
    for s in range(0, len(pos1), batch_size):
        yield pos1[s:s + batch_size]


def iter_candidate_batches(rec, mats, same_country=True, ks=None,
                            cap_per_entity=DEFAULT_CAP_PER_ENTITY, batch_size=DEFAULT_S1_BATCH):
    """Generator: for each S1 batch, build candidates+features and yield the
    assembled df for JUST that batch, then release it. Caller decides what
    to do with each batch (write to disk, subsample+accumulate, score+drop)."""
    ks = ks or DEFAULT_K
    for batch_pos in iter_s1_batches(rec, batch_size):
        pairs = generate_candidates(rec, mats, ks, same_country, cap_per_entity,
                                    s1_positions=batch_pos).reset_index(drop=True)
        if pairs.empty:
            continue
        df = _assemble(rec, pairs, mats)
        del pairs
        yield df
        del df
        gc.collect()


def subsample_negatives_df(df, seed, neg_ratio):
    """Row-level version of train.py's subsample_negatives: keep every
    positive, cap negatives at neg_ratio x (#positives) for THIS df. Used to
    shrink each batch immediately, before it ever gets concatenated with
    the others - this is what keeps the final training table small
    regardless of how many raw candidate pairs blocking produced overall."""
    if not neg_ratio or "y" not in df:
        return df
    pos = df[df["y"] == 1]
    neg = df[df["y"] == 0]
    cap = int(neg_ratio * max(len(pos), 1))
    if len(neg) > cap:
        neg = neg.sample(n=cap, random_state=seed)
    return pd.concat([pos, neg], ignore_index=True)


def build_training_table(rec, truth, same_country=True, ks=None, cap_per_entity=DEFAULT_CAP_PER_ENTITY,
                          batch_size=DEFAULT_S1_BATCH, neg_ratio=20.0, seed=42, cache_dir=None):
    """Batched replacement for `candidates_and_features(...) + df["y"]=...`
    at train time. Processes S1 in chunks, labels each batch against
    `truth`, immediately subsamples that batch's negatives (all positives
    kept, blocking recall for scoring is unaffected - only what gets FIT on
    shrinks), and only accumulates the small, already-subsampled result.
    Peak memory is bounded by one batch's raw candidates, not the whole
    dataset's. If cache_dir is given, each subsampled batch is also written
    to parquet there (so a crash mid-run doesn't lose earlier batches, and
    you can inspect/re-load without recomputation)."""
    mats = build_matrices(rec)
    shards = []
    for k, df in enumerate(iter_candidate_batches(rec, mats, same_country, ks, cap_per_entity, batch_size)):
        df["y"] = [int(c in truth.get(s, ())) for s, c in zip(df["s1_id"], df["cand_id"])]
        df = subsample_negatives_df(df, seed + k, neg_ratio)
        if cache_dir:
            df.to_parquet(f"{cache_dir}/shard_{k:05d}.parquet", index=False)
        shards.append(df)
        print(f"  batch {k}: kept {len(df)} rows after negative subsampling")
    del mats
    gc.collect()
    out = pd.concat(shards, ignore_index=True) if shards else pd.DataFrame()
    del shards
    return out


def predict_in_batches(rec, model, feats, same_country=True, ks=None, cap_per_entity=DEFAULT_CAP_PER_ENTITY,
                        batch_size=DEFAULT_S1_BATCH):
    """Batched replacement for predict.py's single `candidates_and_features`
    + `predict_proba` call. Scores each S1 batch's candidates immediately
    and keeps only (s1_id, cand_id, proba) - not the full feature matrix -
    across batches, so peak memory is one batch's features, not all of
    them. Returns one concatenated (s1_id, cand_id, proba) DataFrame, which
    is all select_matches / macro_f05 need."""
    mats = build_matrices(rec)
    kept = []
    for df in iter_candidate_batches(rec, mats, same_country, ks, cap_per_entity, batch_size):
        proba = model.predict_proba(df[feats])[:, 1]
        kept.append(pd.DataFrame({"s1_id": df["s1_id"].to_numpy(), "cand_id": df["cand_id"].to_numpy(),
                                  "proba": proba}))
        del df, proba
        gc.collect()
    del mats
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=["s1_id", "cand_id", "proba"])


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURES]


def make_model(seed=42, n_jobs=-1):
    try:
        import lightgbm as lgb  # MIT licensed, tiny model - well within the <=8B / MIT-Apache rule
        return lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=n_jobs, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, random_state=seed)


def select_matches(df, thr, exclusive=True):
    """Threshold the probabilities; optionally assign each S2/S3 record to its
    single best S1 entity (S1 is deduplicated)."""
    sel = df[df["proba"] >= thr]
    if exclusive:
        sel = sel.sort_values("proba", ascending=False).drop_duplicates("cand_id")
    return sel.groupby("s1_id")["cand_id"].apply(lambda x: sorted(x)).to_dict()


def macro_f05(s1_ids, truth, pred, beta=0.5):
    """Per-S1 F0.5 averaged over ALL S1 entities; singletons: empty==empty -> 1, else 0."""
    b2 = beta ** 2
    scores = []
    for s in s1_ids:
        t, p = set(truth.get(s, ())), set(pred.get(s, ()))
        if not t and not p:
            scores.append(1.0)
        elif not t or not p:
            scores.append(0.0)
        else:
            tp = len(t & p)
            prec, rec_ = tp / len(p), tp / len(t)
            scores.append(0.0 if tp == 0 else (1 + b2) * prec * rec_ / (b2 * prec + rec_))
    return float(np.mean(scores))