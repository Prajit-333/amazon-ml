"""Train the pair classifier and tune the decision threshold on out-of-fold
predictions using the real metric (macro F0.5 over all Source-1 entities)."""
import argparse
import gc
import json
import os

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from pipeline import (build_training_table, candidates_and_features, feature_columns, load_records,
                      load_truth, macro_f05, make_model, select_matches)


def subsample_negatives(idx, y, seed, neg_ratio):
    """Return a subset of row positions `idx` keeping ALL positives and a
    random sample of negatives capped at neg_ratio x (#positives). This is
    what actually shrinks the LightGBM fit's memory/time footprint - the
    vast majority of candidate pairs are easy, low-similarity negatives that
    add little beyond a representative sample. Validation/OOF predictions
    are NOT subsampled anywhere - only the rows passed to .fit()."""
    if not neg_ratio:
        return idx
    y_idx = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
    pos = idx[y_idx.to_numpy() == 1] if hasattr(y_idx, "to_numpy") else idx[np.asarray(y_idx) == 1]
    neg = idx[~np.isin(idx, pos)]
    cap = int(neg_ratio * max(len(pos), 1))
    if len(neg) > cap:
        rng = np.random.default_rng(seed)
        neg = rng.choice(neg, size=cap, replace=False)
    return np.concatenate([pos, neg])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cross-country", action="store_true",
                    help="do not restrict blocking to the same country label")
    ap.add_argument("--cap-per-entity", type=int, default=200,
                    help="max candidates kept per S1 entity after blocking (0 = no cap)")
    ap.add_argument("--neg-ratio", type=float, default=20.0,
                    help="max negatives per positive used to FIT the model (0 = use all, no subsampling). "
                         "Does not affect blocking recall or OOF scoring - only shrinks .fit() memory/time.")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="threads for the model (try 1 if you suspect an OpenMP DLL conflict on Windows)")
    ap.add_argument("--large-scale", action="store_true", default=True,
                    help="process S1 in batches, subsampling each batch's negatives immediately instead "
                         "of building candidates+features for the whole dataset at once (default: on - "
                         "this is what avoids the OOM crash at multi-million-row scale). Pass "
                         "--no-large-scale to use the original single-shot path for small/debug runs.")
    ap.add_argument("--no-large-scale", dest="large_scale", action="store_false")
    ap.add_argument("--batch-size", type=int, default=20_000,
                    help="S1 rows processed per batch in --large-scale mode")
    ap.add_argument("--cache-dir", default=None,
                    help="if set, write each batch's (already subsampled) shard to parquet here as it "
                         "completes, so a crash mid-run does not lose earlier batches")
    a = ap.parse_args()
    same_country = not a.cross_country
    cap = a.cap_per_entity or None

    rec = load_records(f"{a.data_dir}/train", "train")
    truth = load_truth(f"{a.data_dir}/train/train_ground_truth.tsv")
    s1_ids = rec.loc[rec["src"] == 1, "entity_id"].tolist()

    if a.cache_dir:
        os.makedirs(a.cache_dir, exist_ok=True)

    if a.large_scale:
        # NOTE on evaluation trade-off: negatives are subsampled PER BATCH,
        # before the GroupKFold split below. All positives are always kept
        # (blocking recall is reported exactly), but OOF/threshold-tuning
        # below now scores against a subsampled negative pool rather than
        # every blocked candidate, so the reported F0.5 is a close but not
        # exact estimate of what predict.py's full (unsampled) scoring will
        # see. Raise --neg-ratio (or drop to --no-large-scale on a sample of
        # the data) if you want a tighter OOF estimate before trusting it.
        df = build_training_table(rec, truth, same_country=same_country, cap_per_entity=cap,
                                  batch_size=a.batch_size, neg_ratio=a.neg_ratio, seed=a.seed,
                                  cache_dir=a.cache_dir)
    else:
        df = candidates_and_features(rec, same_country=same_country, cap_per_entity=cap)
        df["y"] = [int(c in truth.get(s, ())) for s, c in zip(df["s1_id"], df["cand_id"])]

    n_true = sum(len(v) for v in truth.values())
    n_single = sum(1 for s in s1_ids if not truth.get(s))
    print(f"S1 entities: {len(s1_ids)} (singletons: {n_single}) | true pairs: {n_true}")
    print(f"candidate pairs: {len(df)} | blocking recall: {df['y'].sum() / max(n_true, 1):.4f} "
          f"| avg cands/S1: {len(df) / len(s1_ids):.1f}")

    feats = feature_columns(df)
    X = df[feats].to_numpy(dtype=np.float32)  # one compact array, reused by every fold/fit -
                                               # avoids a fresh pandas copy per .iloc[...] call
    y_arr = df["y"].to_numpy()
    gc.collect()  # release anything left over from blocking/feature-building before training starts

    oof = np.zeros(len(df))
    for k, (tr, va) in enumerate(GroupKFold(a.folds).split(df, df["y"], df["s1_id"])):
        fit_idx = subsample_negatives(tr, df["y"], a.seed + k, a.neg_ratio)
        print(f"fold {k}: fitting on {len(fit_idx)} pairs (of {len(tr)} available)")
        m = make_model(a.seed, n_jobs=a.n_jobs)
        m.fit(X[fit_idx], y_arr[fit_idx])
        oof[va] = m.predict_proba(X[va])[:, 1]
        del m
        print(f"fold {k} done")
    df["proba"] = oof

    best = (-1.0, 0.5, True)
    for exclusive in (False, True):
        for thr in np.arange(0.30, 0.96, 0.025):
            f = macro_f05(s1_ids, truth, select_matches(df, thr, exclusive))
            if f > best[0]:
                best = (f, float(thr), exclusive)
        print(f"exclusive={exclusive}: best so far F0.5={best[0]:.4f}")
    print(f"BEST OOF macro F0.5 = {best[0]:.4f} at thr={best[1]:.3f}, exclusive={best[2]}")

    final_idx = subsample_negatives(np.arange(len(df)), df["y"], a.seed, a.neg_ratio)
    print(f"final fit on {len(final_idx)} pairs (of {len(df)} available)")
    final = make_model(a.seed, n_jobs=a.n_jobs)
    final.fit(X[final_idx], y_arr[final_idx])
    os.makedirs(a.model_dir, exist_ok=True)
    joblib.dump({"model": final, "features": feats, "thr": best[1], "exclusive": best[2],
                 "same_country": same_country, "cap_per_entity": cap}, f"{a.model_dir}/matcher.joblib")
    with open(f"{a.model_dir}/meta.json", "w") as f:
        json.dump({"oof_f05": best[0], "thr": best[1], "exclusive": best[2],
                   "blocking_recall": float(df["y"].sum() / max(n_true, 1))}, f, indent=2)
    print("saved model")


if __name__ == "__main__":
    main()