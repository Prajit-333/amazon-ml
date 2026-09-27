"""Phase 1: audit raw data, learn normalization examples, and persist records.

The raw columns are retained beside derived normalized columns. The pickle is the
machine-readable checkpoint; normalized_sample.tsv is for human inspection.
"""
import argparse
import csv
import json
import os

import pandas as pd

from pipeline import load_truth
from normalize import prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="artifacts/phase1")
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--audit-sample", type=int, default=200)
    ap.add_argument("--chunk-rows", type=int, default=100000,
                    help="rows per persisted normalization chunk")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="limit rows per source for a quick smoke run; 0 means all rows")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    chunk_dir = os.path.join(args.out_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_paths = []
    sample_parts = []
    source_counts = {}
    country_values = set()
    missing_raw = {c: 0 for c in ("business_name", "business_address", "country")}
    missing_normalized = {c: 0 for c in ("name_core", "addr_core", "postal", "ctry")}
    total_rows = 0
    for source in (1, 2, 3):
        path = f"{args.data_dir}/{args.split}/{args.split}_source{source}.tsv"
        reader = pd.read_csv(path, sep="\t", dtype=object, keep_default_na=False,
                              quoting=csv.QUOTE_NONE, chunksize=args.chunk_rows)
        source_total = 0
        for chunk_number, raw in enumerate(reader):
            if args.max_rows and source_total >= args.max_rows:
                break
            if args.max_rows:
                raw = raw.head(args.max_rows - source_total).copy()
            raw["src"] = source
            normalized = prepare(raw)
            chunk_path = os.path.join(chunk_dir, f"source{source}_{chunk_number:05d}.pkl")
            normalized.to_pickle(chunk_path)
            chunk_paths.append(chunk_path)
            source_total += len(normalized)
            total_rows += len(normalized)
            source_counts[str(source)] = source_total
            for column in missing_raw:
                missing_raw[column] += int(raw[column].eq("").sum())
            for column in missing_normalized:
                missing_normalized[column] += int(normalized[column].eq("").sum())
            country_values.update(normalized["ctry"].drop_duplicates().tolist())
            if sum(len(part) for part in sample_parts) < args.audit_sample:
                sample_parts.append(normalized.head(args.audit_sample - sum(len(part) for part in sample_parts)))
            print(f"phase1: wrote {chunk_path} ({len(normalized)} rows)", flush=True)
            del raw, normalized

    sample = pd.concat(sample_parts, ignore_index=True).head(args.audit_sample)
    sample.to_csv(f"{args.out_dir}/normalized_sample.tsv", sep="\t", index=False)
    if args.max_rows:
        sample.to_pickle(f"{args.out_dir}/records_normalized.pkl")

    audit = {
        "split": args.split,
        "rows": int(total_rows),
        "chunk_rows": args.chunk_rows,
        "chunk_files": chunk_paths,
        "source_counts": source_counts,
        "country_values": sorted(country_values),
        "missing_raw": missing_raw,
        "missing_normalized": missing_normalized,
        "normalization_columns": [
            "name_core", "name_legal", "name_full", "addr_core", "addr_full",
            "postal", "nums", "ctry", "text_full",
        ],
        "audit_examples": sample[[
            "entity_id", "business_name", "name_full", "name_core",
            "business_address", "addr_full", "addr_core", "postal", "country", "ctry",
        ]].fillna("").to_dict(orient="records"),
    }
    with open(f"{args.out_dir}/audit.json", "w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2, ensure_ascii=False)

    if args.split == "train":
        truth_path = f"{args.data_dir}/train/train_ground_truth.tsv"
        truth = load_truth(truth_path)
        audit["ground_truth_rows"] = len(truth)
        audit["ground_truth_sample"] = [
            {"source1_entity_id": key, "matched_entity_ids": sorted(values)}
            for key, values in list(truth.items())[:args.audit_sample]
        ]
        with open(f"{args.out_dir}/audit.json", "w", encoding="utf-8") as file:
            json.dump(audit, file, indent=2, ensure_ascii=False)

    print(f"phase1 complete: {total_rows} records in {len(chunk_paths)} chunks")
    print(f"inspect: {args.out_dir}/audit.json and {args.out_dir}/normalized_sample.tsv")


if __name__ == "__main__":
    main()
