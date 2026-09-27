"""Phase 4: evaluate saved candidate pairs against ground truth."""
import argparse
import glob
import json
import os

import pandas as pd

from pipeline import load_truth, read_tsv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="artifacts/phase3/candidate_pairs.tsv")
    ap.add_argument("--ground-truth", default="dataset/train/train_ground_truth.tsv")
    ap.add_argument("--records", default="artifacts/phase2/records_features.pkl")
    ap.add_argument("--records-dir", default=None,
                    help="Phase 2 directory containing feature chunks")
    ap.add_argument("--out-dir", default="artifacts/phase4")
    ap.add_argument("--limit-s1", type=int, default=0,
                    help="evaluate only the first N S1 IDs; 0 means all")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    candidates = pd.read_csv(args.candidates, sep="\t", dtype={"s1_id": str, "cand_id": str})
    truth = load_truth(args.ground_truth)
    if args.records_dir:
        paths = sorted(glob.glob(os.path.join(args.records_dir, "chunks", "*.pkl")))
        if not paths:
            raise FileNotFoundError("No Phase 2 feature chunks found")
        s1_ids = []
        for path in paths:
            records = pd.read_pickle(path)
            s1_ids.extend(records.loc[records["src"] == 1, "entity_id"].astype(str).tolist())
            del records
    else:
        records = pd.read_pickle(args.records)
        s1_ids = records.loc[records["src"] == 1, "entity_id"].astype(str).tolist()
    if args.limit_s1:
        s1_ids = s1_ids[:args.limit_s1]
    allowed = set(s1_ids)
    candidates = candidates[candidates["s1_id"].isin(allowed)]

    candidate_map = candidates.groupby("s1_id")["cand_id"].apply(set).to_dict()
    rows = []
    covered = 0
    truth_pairs = 0
    found_pairs = 0
    missing_rows = []
    for s1_id in s1_ids:
        actual = set(truth.get(s1_id, ()))
        found = candidate_map.get(s1_id, set())
        missed = sorted(actual - found)
        hit = len(actual & found)
        truth_pairs += len(actual)
        found_pairs += hit
        covered += int(bool(actual) and hit == len(actual))
        rows.append({
            "s1_id": s1_id,
            "truth_count": len(actual),
            "candidate_count": len(found),
            "found_truth_count": hit,
            "recall": hit / len(actual) if actual else 1.0,
            "all_truth_found": not missed,
        })
        for cand_id in missed:
            missing_rows.append({"s1_id": s1_id, "missed_cand_id": cand_id})

    report = pd.DataFrame(rows)
    report.to_csv(f"{args.out_dir}/per_s1_recall.tsv", sep="\t", index=False)
    pd.DataFrame(missing_rows, columns=["s1_id", "missed_cand_id"]).to_csv(
        f"{args.out_dir}/missed_truth_pairs.tsv", sep="\t", index=False)
    summary = {
        "s1_evaluated": len(s1_ids),
        "s1_with_truth": int(sum(bool(truth.get(s)) for s in s1_ids)),
        "s1_with_all_truth_candidates": int(covered),
        "truth_pairs": truth_pairs,
        "found_truth_pairs": found_pairs,
        "pair_recall": found_pairs / max(truth_pairs, 1),
        "all_truth_per_s1_recall": covered / max(sum(bool(truth.get(s)) for s in s1_ids), 1),
        "candidate_pairs_evaluated": int(len(candidates)),
    }
    with open(f"{args.out_dir}/summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"inspect: {args.out_dir}/per_s1_recall.tsv and {args.out_dir}/missed_truth_pairs.tsv")


if __name__ == "__main__":
    main()
