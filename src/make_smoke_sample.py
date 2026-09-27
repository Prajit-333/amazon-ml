"""Build a SMALL but referentially-consistent sample for smoke-testing the
pipeline: pick a handful of Source-1 entities, then pull in every Source-2/3
record that is actually a true match for them (per ground truth), plus a
random pad of extra Source-2/3 rows so blocking has some noise to chew on.

Unlike `head -n N` on each file independently, this guarantees blocking has
a real chance of finding true matches, so recall/F0.5 on the sample are
actually meaningful (not just "did it crash").
"""
import argparse
import csv
import os
import random

import pandas as pd

from pipeline import read_tsv  # reuse the same safe TSV reader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="dataset_small")
    ap.add_argument("--n-s1", type=int, default=2000, help="number of S1 entities to sample")
    ap.add_argument("--pad", type=int, default=3000, help="extra random S2/S3 rows to add as noise")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    s1 = read_tsv(f"{a.data_dir}/train/train_source1.tsv")
    s2 = read_tsv(f"{a.data_dir}/train/train_source2.tsv")
    s3 = read_tsv(f"{a.data_dir}/train/train_source3.tsv")
    gt = read_tsv(f"{a.data_dir}/train/train_ground_truth.tsv")

    gt_map = dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"]))

    # sample S1 entities, biased toward ones that actually have a match so the
    # smoke test exercises real positives, not just singletons
    ids_with_match = [i for i in s1["entity_id"] if gt_map.get(i, "").strip()]
    ids_without = [i for i in s1["entity_id"] if i not in set(ids_with_match)]
    rng.shuffle(ids_with_match)
    rng.shuffle(ids_without)
    n_with = min(len(ids_with_match), a.n_s1 // 2)
    keep_s1_ids = set(ids_with_match[:n_with] + ids_without[:a.n_s1 - n_with])

    s1_small = s1[s1["entity_id"].isin(keep_s1_ids)].reset_index(drop=True)

    # every S2/S3 id that is a true match for a kept S1 entity - these MUST
    # be present or blocking_recall is meaningless by construction
    needed_ids = set()
    for i in keep_s1_ids:
        needed_ids.update(x.strip() for x in gt_map.get(i, "").split(",") if x.strip())

    def build_split(df, needed):
        must_keep = df[df["entity_id"].isin(needed)]
        rest = df[~df["entity_id"].isin(needed)]
        pad = rest.sample(n=min(a.pad, len(rest)), random_state=a.seed)
        return pd.concat([must_keep, pad], ignore_index=True)

    s2_small = build_split(s2, needed_ids)
    s3_small = build_split(s3, needed_ids)

    gt_small = gt[gt["source1_entity_id"].isin(keep_s1_ids)]

    os.makedirs(f"{a.out_dir}/train", exist_ok=True)
    s1_small.to_csv(f"{a.out_dir}/train/train_source1.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)
    s2_small.to_csv(f"{a.out_dir}/train/train_source2.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)
    s3_small.to_csv(f"{a.out_dir}/train/train_source3.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)
    gt_small.to_csv(f"{a.out_dir}/train/train_ground_truth.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE)

    print(f"S1: {len(s1_small)} | S2: {len(s2_small)} | S3: {len(s3_small)} | "
          f"S1 with a true match: {n_with} | ground truth rows: {len(gt_small)}")


if __name__ == "__main__":
    main()