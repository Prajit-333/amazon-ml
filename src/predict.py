"""Run the trained matcher on the test set and write
output/matching_results.tsv and output/candidate_pairs.tsv."""
import argparse
import os

import joblib

from src.pipeline import candidates_and_features, load_records, predict_in_batches, select_matches


def write_tsv(path, header, s1_ids, mapping):
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for s in s1_ids:
            f.write(f"{s}\t{','.join(mapping.get(s, []))}\n")  # empty list -> empty cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--threshold", type=float, default=None, help="override tuned threshold")
    ap.add_argument("--large-scale", action="store_true", default=True,
                    help="score S1 in batches, keeping only (s1_id, cand_id, proba) across batches "
                         "instead of holding every candidate's full feature row for the whole test set "
                         "at once. Pass --no-large-scale for the original single-shot path.")
    ap.add_argument("--no-large-scale", dest="large_scale", action="store_false")
    ap.add_argument("--batch-size", type=int, default=20_000)
    a = ap.parse_args()

    bundle = joblib.load(f"{a.model_dir}/matcher.joblib")
    rec = load_records(f"{a.data_dir}/test", "test")
    s1_ids = rec.loc[rec["src"] == 1, "entity_id"].tolist()

    if a.large_scale:
        df = predict_in_batches(rec, bundle["model"], bundle["features"], same_country=bundle["same_country"],
                                cap_per_entity=bundle.get("cap_per_entity"), batch_size=a.batch_size)
    else:
        df = candidates_and_features(rec, same_country=bundle["same_country"])
        df["proba"] = bundle["model"].predict_proba(df[bundle["features"]])[:, 1] if len(df) else []
    thr = a.threshold if a.threshold is not None else bundle["thr"]

    pred = select_matches(df, thr, bundle["exclusive"])
    cands = df.groupby("s1_id")["cand_id"].apply(lambda x: sorted(set(x))).to_dict()

    os.makedirs(a.out_dir, exist_ok=True)
    write_tsv(f"{a.out_dir}/matching_results.tsv", ["source1_entity_id", "matched_entity_ids"], s1_ids, pred)
    write_tsv(f"{a.out_dir}/candidate_pairs.tsv", ["source1_entity_id", "candidate_entity_ids"], s1_ids, cands)

    # sanity checks (the official utils/validate_submission.py is still the reference)
    all_other = set(rec.loc[rec["src"] != 1, "entity_id"])
    for s, ids in pred.items():
        assert len(ids) == len(set(ids)), f"duplicate ids for {s}"
        assert set(ids) <= all_other, f"unknown ids for {s}"
        assert set(ids) <= set(cands.get(s, [])), f"match not in candidates for {s}"
    n_match = sum(1 for s in s1_ids if pred.get(s))
    print(f"wrote {len(s1_ids)} rows | S1 with >=1 match: {n_match} | singletons predicted: {len(s1_ids) - n_match}")
    print(rec.loc[rec['src'] == 1, 'ctry'].value_counts().to_string())


if __name__ == "__main__":
    main()