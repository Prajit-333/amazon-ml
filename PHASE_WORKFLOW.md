# Inspectable ML workflow

Run every command from the repository root (`F:\entity relation ml`). Each phase writes a checkpoint and can be inspected before the next phase.

## Phase 1: audit and normalization

```powershell
python src\phase1_audit_normalize.py --data-dir dataset --out-dir artifacts\phase1 --split train --audit-sample 200
```

Inspect:

- `artifacts/phase1/audit.json`
- `artifacts/phase1/normalized_sample.tsv`
- `artifacts/phase1/records_normalized.pkl`

Raw fields remain in the checkpoint next to normalized fields. The 200 examples are an audit sample, not a learned model; the current rulebook is explicit in `src/normalize.py`.

For a smoke run, add `--max-rows 2000`.

## Phase 2: record features

```powershell
python src\phase2_features.py --input-dir artifacts\phase1 --out-dir artifacts\phase2
```

Inspect `features_sample.tsv` and `numeric_summary.tsv`. This phase does not compare records with one another.

## Phase 3: candidate generation only

```powershell
python src\phase3_candidates.py --input-dir artifacts\phase2 --out-dir artifacts\phase3 --batch-size 50000 --target-chunk-size 500000 --cap-per-entity 100
```

Inspect:

- `artifacts/phase3/candidate_pairs.tsv`
- `artifacts/phase3/summary.json`

This phase runs the existing TF-IDF, token, phonetic, postal, address-key, and fallback blocks. Duplicate `(S1, candidate)` rows are collapsed by the blocking implementation.

## Phase 4: candidate recall gate

```powershell
python src\phase4_recall.py --candidates artifacts\phase3\candidate_pairs.tsv --records-dir artifacts\phase2 --ground-truth dataset\train\train_ground_truth.tsv --out-dir artifacts\phase4
```

For a quick subset, add `--limit-s1 200`. Inspect `summary.json`, `per_s1_recall.tsv`, and `missed_truth_pairs.tsv`. Do not train until pair recall is acceptable.

## Phase 5: feature computation and training

```powershell
python src\phase5_train.py --records artifacts\phase2\records_features.pkl --candidates artifacts\phase3\candidate_pairs.tsv --ground-truth dataset\train\train_ground_truth.tsv --model-dir models --neg-ratio 20 --n-jobs 1
```

Only after the candidate recall gate passes does this phase compute pairwise fuzzy/cosine/competition features, split by S1 group, subsample negatives for fitting, cross-validate, and save `models/matcher.joblib` and `models/meta.json`.

## Current limitations

- The normalization rulebook is hand-authored. Phase 1 audits 200 ground-truth examples but does not automatically infer new rules. Add a reviewed rule to `src/normalize.py`, rerun Phase 1, and compare audit output.
- Phase 3 currently persists candidate pairs as TSV for portability. For very large candidate tables, use a database or Parquet engine after installing a Parquet dependency.
- Phase 4 is a gate, not an automatic block-tuning loop. This makes each change auditable: change blocking parameters or add a block, rerun Phase 3, then compare Phase 4 reports.
