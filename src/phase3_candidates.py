"""Phase 3: generate and persist candidate pairs only.

No pairwise fuzzy features and no model fitting happen here. The output is a
reviewable candidate table that Phase 4 can score against ground truth.
"""
import argparse
import gc
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from blocking import (
    DEFAULT_CAP_PER_ENTITY, DEFAULT_K, REPS, _aggregate_iteration, _index_pass_chunk,
    build_indexes, build_matrices, generate_candidates, topk_rows,
)
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer


def generate_index_candidates(records, positions, batch_size, cap_per_entity, same_country):
    """Generate candidates with bounded inverted-index blocks only.

    This avoids the multi-gigabyte global TF-IDF matrices required by the
    default path. It retains token, phonetic, postal, and address-key blocks.
    """
    source = records["src"].to_numpy()
    countries = records["ctry"].to_numpy()
    name_core = records["name_core"].to_numpy()
    addr_core = records["addr_core"].to_numpy()
    postal = records["postal"].to_numpy()
    nums = records["nums"].to_numpy()
    size = len(records)
    for batch_number, start in enumerate(range(0, len(positions), batch_size)):
        batch = positions[start:start + batch_size]
        results = []
        groups = sorted(set(countries[batch])) if same_country else [None]
        for target_source in (2, 3):
            target_positions = np.where(source == target_source)[0]
            for country in groups:
                s_mask = np.zeros(size, dtype=bool)
                t_mask = np.zeros(size, dtype=bool)
                s_mask[batch] = True
                t_mask[target_positions] = True
                if country is not None:
                    s_mask &= countries == country
                    t_mask &= countries == country
                s_idx = np.where(s_mask)[0]
                if not len(s_idx) or not t_mask.any():
                    continue
                indexes = build_indexes(records, t_mask)
                raw = _index_pass_chunk(s_idx, name_core, addr_core, postal, nums, *indexes)
                results.append(_aggregate_iteration([raw], cap_per_entity))
                del indexes, raw
        if results:
            pairs = pd.concat(results, ignore_index=True)
            pairs = (pairs.sort_values(["i", "n_strategies", "block_score"],
                                       ascending=[True, False, False])
                     .groupby("i", sort=False).head(cap_per_entity or len(pairs))
                     .reset_index(drop=True))
            yield pairs
        del results
        gc.collect()


def generate_tfidf_candidates(records, positions, batch_size, target_chunk_size,
                              cap_per_entity, same_country, progress_path=None):
    """Generate TF-IDF candidates without retaining a global matrix.

    Each S1/country/source batch is transformed once per representation, while
    target records are transformed in bounded chunks. The TF-IDF strategy is
    therefore preserved, but sparse matrices never contain the full corpus.
    """
    source = records["src"].to_numpy()
    countries = records["ctry"].to_numpy()
    size = len(records)
    started = time.time()
    total_batches = (len(positions) + batch_size - 1) // batch_size
    completed_chunks = 0
    total_chunks = 0
    for target_source in (2, 3):
        target_positions = np.where(source == target_source)[0]
        for country in (sorted(set(countries[positions])) if same_country else [None]):
            t_count = len(target_positions[countries[target_positions] == country]) if country is not None else len(target_positions)
            total_chunks += (t_count + target_chunk_size - 1) // target_chunk_size
    total_chunks *= total_batches

    for batch_number, start in enumerate(range(0, len(positions), batch_size)):
        batch = positions[start:start + batch_size]
        results = []
        groups = sorted(set(countries[batch])) if same_country else [None]
        for target_source in (2, 3):
            target_positions = np.where(source == target_source)[0]
            for country in groups:
                s_idx = batch[countries[batch] == country] if country is not None else batch
                t_idx = target_positions[countries[target_positions] == country] if country is not None else target_positions
                if not len(s_idx) or not len(t_idx):
                    continue
                for rep, k in DEFAULT_K.items():
                    column, options = REPS[rep]
                    options = dict(options)
                    n_features = options.pop("n_features")
                    vectorizer = HashingVectorizer(dtype=np.float32, alternate_sign=False,
                                                   n_features=n_features, **options)
                    source_counts = vectorizer.transform(records[column].iloc[s_idx])
                    source_matrix = TfidfTransformer(sublinear_tf=True, use_idf=False).fit_transform(source_counts)
                    del source_counts
                    for target_number, target_start in enumerate(range(0, len(t_idx), target_chunk_size)):
                        target_rows = t_idx[target_start:target_start + target_chunk_size]
                        target_counts = vectorizer.transform(records[column].iloc[target_rows])
                        target_matrix = TfidfTransformer(sublinear_tf=True, use_idf=False).fit_transform(target_counts)
                        idx, sim = topk_rows(source_matrix, target_matrix, k)
                        keep = sim.ravel() > 0
                        if keep.any():
                            raw = pd.DataFrame({
                                "i": np.repeat(s_idx, idx.shape[1])[keep],
                                "j": target_rows[idx.ravel()][keep],
                                "strategy": f"tfidf:{rep}",
                                "score": sim.ravel()[keep],
                            })
                            results.append(_aggregate_iteration([raw], cap_per_entity))
                        print(
                            f"phase3 batch {batch_number + 1}/{total_batches}: {rep} "
                            f"target chunk {target_number + 1}/{(len(t_idx) + target_chunk_size - 1) // target_chunk_size}",
                            flush=True,
                        )
                        completed_chunks += 1
                        progress = {
                            "batch": batch_number + 1,
                            "batches_total": total_batches,
                            "representation": rep,
                            "target_source": target_source,
                            "target_chunk": target_number + 1,
                            "target_chunks_total": (len(t_idx) + target_chunk_size - 1) // target_chunk_size,
                            "chunks_completed": completed_chunks,
                            "chunks_total_estimate": total_chunks,
                            "percent_estimate": 100.0 * completed_chunks / max(total_chunks, 1),
                            "elapsed_seconds": time.time() - started,
                        }
                        if progress_path:
                            with open(progress_path, "w", encoding="utf-8") as file:
                                json.dump(progress, file, indent=2)
                        del target_counts, target_matrix, idx, sim
                    del source_matrix, vectorizer
        if results:
            pairs = _aggregate_iteration(
                [result.assign(strategy="aggregated", score=result["block_score"])
                 for result in results], cap_per_entity)
            yield pairs
        del results
        gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="artifacts/phase2/records_features.pkl")
    ap.add_argument("--input-dir", default=None,
                    help="Phase 2 directory containing feature chunks")
    ap.add_argument("--out-dir", default="artifacts/phase3")
    ap.add_argument("--batch-size", type=int, default=20000)
    ap.add_argument("--cap-per-entity", type=int, default=DEFAULT_CAP_PER_ENTITY)
    ap.add_argument("--cross-country", action="store_true")
    ap.add_argument("--index-only", action="store_true",
                    help="avoid global TF-IDF matrices; use token/phonetic/postal/address blocks")
    ap.add_argument("--target-chunk-size", type=int, default=100000,
                    help="target rows per TF-IDF comparison chunk")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.json")

    input_paths = sorted(glob.glob(os.path.join(args.input_dir, "chunks", "*.pkl"))) if args.input_dir else [args.input]
    if not input_paths:
        raise FileNotFoundError("No Phase 2 feature chunks found")
    records = pd.concat((pd.read_pickle(path) for path in input_paths), ignore_index=True)
    positions = records.index[records["src"].to_numpy() == 1].to_numpy()
    output_path = f"{args.out_dir}/candidate_pairs.tsv"
    first = True
    total = 0

    if args.index_only:
        pair_batches = generate_index_candidates(
            records, positions, args.batch_size, args.cap_per_entity or None,
            not args.cross_country)
    else:
        pair_batches = generate_tfidf_candidates(
            records, positions, args.batch_size, args.target_chunk_size,
            args.cap_per_entity or None, not args.cross_country, progress_path
        )

    for batch_number, pairs in enumerate(pair_batches):
        if pairs.empty:
            continue
        ids = records["entity_id"].to_numpy()
        pairs.insert(0, "s1_id", ids[pairs["i"].to_numpy()])
        pairs.insert(1, "cand_id", ids[pairs["j"].to_numpy()])
        pairs["candidate_src"] = records["src"].to_numpy()[pairs["j"].to_numpy()]
        pairs.to_csv(output_path, sep="\t", index=False, mode="w" if first else "a", header=first)
        first = False
        total += len(pairs)
        print(f"phase3 batch {batch_number}: wrote {len(pairs)} candidate pairs", flush=True)
        del pairs
        gc.collect()

    with open(progress_path, "w", encoding="utf-8") as file:
        json.dump({"status": "complete", "candidate_pairs": total}, file, indent=2)

    summary = {
        "records": int(len(records)),
        "s1_records": int(len(positions)),
        "candidate_pairs": int(total),
        "candidate_file": output_path,
        "same_country": not args.cross_country,
        "index_only": args.index_only,
        "cap_per_entity": args.cap_per_entity,
        "blocking_parameters": {"DEFAULT_K": DEFAULT_K},
    }
    with open(f"{args.out_dir}/summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"phase3 complete: {total} candidate pairs")
    print(f"inspect: {output_path} and {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
