"""Phase 5: build pair features from saved candidates and train the matcher."""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from features import add_competition, build_features
from pipeline import feature_columns, load_truth, make_model, macro_f05, select_matches
from blocking import build_matrices


def sample_indices(indices, y, seed, neg_ratio):
    if not neg_ratio:
        return indices
    positives = indices[y[indices] == 1]
    negatives = indices[y[indices] == 0]
    cap = int(neg_ratio * max(len(positives), 1))
    if len(negatives) > cap:
        negatives = np.random.default_rng(seed).choice(negatives, cap, replace=False)
    return np.concatenate([positives, negatives])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="artifacts/phase2/records_features.pkl")
    ap.add_argument("--candidates", default="artifacts/phase3/candidate_pairs.tsv")
    ap.add_argument("--ground-truth", default="dataset/train/train_ground_truth.tsv")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--neg-ratio", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    records = pd.read_pickle(args.records)
    pairs = pd.read_csv(args.candidates, sep="\t")
    truth = load_truth(args.ground_truth)
    ids = records["entity_id"].to_numpy()
    pairs["s1_id"] = ids[pairs["i"].to_numpy()]
    pairs["cand_id"] = ids[pairs["j"].to_numpy()]
    pairs["y"] = np.asarray([int(c in truth.get(s, ()))
                              for s, c in zip(pairs["s1_id"], pairs["cand_id"])], dtype=np.int8)

    print(f"phase5: loaded {len(pairs)} saved candidates")
    print(f"phase5: positives={int(pairs['y'].sum())}, negatives={int((pairs['y'] == 0).sum())}")
    mats = build_matrices(records)
    feature_frame = build_features(records, pairs, mats)
    df = pd.concat([pairs.reset_index(drop=True), feature_frame], axis=1)
    df = add_competition(df)
    del feature_frame, mats

    feats = feature_columns(df)
    X = df[feats].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy()
    s1_ids = records.loc[records["src"] == 1, "entity_id"].tolist()
    oof = np.zeros(len(df), dtype=np.float32)
    folds = min(args.folds, max(2, df["s1_id"].nunique()))
    for fold, (tr, va) in enumerate(GroupKFold(folds).split(X, y, df["s1_id"])):
        fit_idx = sample_indices(tr, y, args.seed + fold, args.neg_ratio)
        print(f"phase5 fold {fold}: fitting on {len(fit_idx)} pairs (of {len(tr)} available)")
        model = make_model(args.seed, n_jobs=args.n_jobs)
        model.fit(X[fit_idx], y[fit_idx])
        oof[va] = model.predict_proba(X[va])[:, 1]

    df["proba"] = oof
    best = (-1.0, 0.5, True)
    for exclusive in (False, True):
        for threshold in np.arange(0.30, 0.96, 0.025):
            score = macro_f05(s1_ids, truth, select_matches(df, threshold, exclusive))
            if score > best[0]:
                best = (score, float(threshold), exclusive)
        print(f"phase5 exclusive={exclusive}: best F0.5={best[0]:.4f}")

    final_idx = sample_indices(np.arange(len(df)), y, args.seed, args.neg_ratio)
    print(f"phase5 final fit: {len(final_idx)} pairs (of {len(df)} available)")
    final = make_model(args.seed, n_jobs=args.n_jobs)
    final.fit(X[final_idx], y[final_idx])
    joblib.dump({"model": final, "features": feats, "thr": best[1],
                 "exclusive": best[2], "same_country": True},
                f"{args.model_dir}/matcher.joblib")
    with open(f"{args.model_dir}/meta.json", "w", encoding="utf-8") as file:
        json.dump({"oof_f05": best[0], "thr": best[1], "exclusive": best[2],
                   "candidate_pairs": len(df), "positive_pairs": int(y.sum()),
                   "negative_pairs": int((y == 0).sum())}, file, indent=2)
    print("phase5 complete: saved model")


if __name__ == "__main__":
    main()
