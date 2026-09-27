"""Phase 2: derive record-level features from the Phase 1 checkpoint."""
import argparse
import os
import glob

import numpy as np
import pandas as pd


def add_record_features(records):
    out = records.copy()
    for column in ("name_core", "name_full", "addr_core", "addr_full", "postal", "nums", "ctry"):
        out[column] = out[column].fillna("").astype(str)

    out["name_len"] = out["name_core"].str.len().astype(np.int32)
    out["name_tokens"] = out["name_core"].str.split().str.len().astype(np.int16)
    out["name_digits"] = out["name_core"].str.count(r"\d").astype(np.int16)
    out["address_len"] = out["addr_core"].str.len().astype(np.int32)
    out["address_tokens"] = out["addr_core"].str.split().str.len().astype(np.int16)
    out["address_digits"] = out["addr_core"].str.count(r"\d").astype(np.int16)
    out["postal_present"] = out["postal"].ne("").astype(np.int8)
    out["postal_prefix"] = out["postal"].str[:3]
    out["postal_length"] = out["postal"].str.len().astype(np.int8)
    out["has_house_number"] = out["nums"].ne("").astype(np.int8)
    out["name_initial"] = out["name_core"].str[:1]
    out["address_initial"] = out["addr_core"].str[:1]
    out["name_address_len"] = (out["name_len"] + out["address_len"]).astype(np.int32)
    return out


def numeric_columns():
    return [
        "name_len", "name_tokens", "name_digits", "address_len", "address_tokens",
        "address_digits", "postal_present", "postal_length", "has_house_number",
        "name_address_len",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="artifacts/phase1/records_normalized.pkl")
    ap.add_argument("--input-dir", default=None,
                    help="Phase 1 directory containing chunk files")
    ap.add_argument("--out-dir", default="artifacts/phase2")
    ap.add_argument("--sample", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    input_paths = sorted(glob.glob(os.path.join(args.input_dir, "chunks", "*.pkl"))) if args.input_dir else [args.input]
    if not input_paths:
        raise FileNotFoundError("No Phase 1 normalized chunks found")
    feature_dir = os.path.join(args.out_dir, "chunks")
    os.makedirs(feature_dir, exist_ok=True)
    sample_parts = []
    total_rows = 0
    summary_parts = []
    for number, input_path in enumerate(input_paths):
        records = pd.read_pickle(input_path)
        features = add_record_features(records)
        output_path = os.path.join(feature_dir, os.path.basename(input_path))
        features.to_pickle(output_path)
        total_rows += len(features)
        if sum(len(part) for part in sample_parts) < args.sample:
            sample_parts.append(features.head(args.sample - sum(len(part) for part in sample_parts)))
        summary_parts.append(features[numeric_columns()].describe().T)
        print(f"phase2: wrote {output_path} ({len(features)} rows)", flush=True)
        del records, features

    sample = pd.concat(sample_parts, ignore_index=True).head(args.sample)
    sample.to_csv(f"{args.out_dir}/features_sample.tsv", sep="\t", index=False)

    pd.concat(summary_parts).groupby(level=0).mean().reindex(numeric_columns()).to_csv(
        f"{args.out_dir}/numeric_summary.tsv", sep="\t")
    with open(f"{args.out_dir}/manifest.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(os.path.join(feature_dir, os.path.basename(path)) for path in input_paths))
    print(f"phase2 complete: {total_rows} records in {len(input_paths)} chunks")
    print(f"inspect: {args.out_dir}/features_sample.tsv and {args.out_dir}/numeric_summary.tsv")


if __name__ == "__main__":
    main()
