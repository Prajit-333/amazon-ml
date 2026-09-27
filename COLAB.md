# Google Colab workflow

This repository intentionally tracks code and configuration only. The full dataset, generated checkpoints, candidate pairs, models, and logs are ignored by Git because they are large and/or machine-specific.

## 1. Clone the repository

In a Colab cell:

```python
!git clone <YOUR_REPOSITORY_URL> /content/entity-relation-ml
%cd /content/entity-relation-ml
!pip install -r requirements.txt
```

Replace `<YOUR_REPOSITORY_URL>` with the repository URL after pushing the code.

## 2. Make the dataset available

The dataset is not committed to Git. Use Google Drive for the large TSV files:

```python
from google.colab import drive
drive.mount('/content/drive')
```

The expected layout is:

```text
/content/drive/MyDrive/entity-relation-ml/dataset/
  train/train_source1.tsv
  train/train_source2.tsv
  train/train_source3.tsv
  train/train_ground_truth.tsv
  test/test_source1.tsv
  test/test_source2.tsv
  test/test_source3.tsv
```

Point the phase commands at that directory. Keeping checkpoints under `/content/drive` prevents losing them when the Colab runtime resets.

## 3. Run the phases in order

### Phase 1: audit and normalization

```python
!python src/phase1_audit_normalize.py \
  --data-dir /content/drive/MyDrive/entity-relation-ml/dataset \
  --out-dir /content/drive/MyDrive/entity-relation-ml/artifacts/phase1 \
  --split train \
  --audit-sample 200 \
  --chunk-rows 100000
```

Inspect `audit.json` and `normalized_sample.tsv`. The raw columns remain beside normalized columns in every checkpoint chunk.

### Phase 2: record features

```python
!python src/phase2_features.py \
  --input-dir /content/drive/MyDrive/entity-relation-ml/artifacts/phase1 \
  --out-dir /content/drive/MyDrive/entity-relation-ml/artifacts/phase2 \
  --sample 200
```

### Phase 3: TF-IDF candidate generation

This keeps TF-IDF enabled while bounding memory. `progress.json` is updated after each target chunk.

```python
!python src/phase3_candidates.py \
  --input-dir /content/drive/MyDrive/entity-relation-ml/artifacts/phase2 \
  --out-dir /content/drive/MyDrive/entity-relation-ml/artifacts/phase3 \
  --batch-size 50000 \
  --target-chunk-size 500000 \
  --cap-per-entity 100
```

Inspect:

```text
artifacts/phase3/progress.json
artifacts/phase3/candidate_pairs.tsv
artifacts/phase3/summary.json
```

### Phase 4: candidate recall

```python
!python src/phase4_recall.py \
  --candidates /content/drive/MyDrive/entity-relation-ml/artifacts/phase3/candidate_pairs.tsv \
  --records-dir /content/drive/MyDrive/entity-relation-ml/artifacts/phase2 \
  --ground-truth /content/drive/MyDrive/entity-relation-ml/dataset/train/train_ground_truth.tsv \
  --out-dir /content/drive/MyDrive/entity-relation-ml/artifacts/phase4
```

Do not train until candidate recall is acceptable.

### Phase 5: train

```python
!python src/phase5_train.py \
  --records /content/drive/MyDrive/entity-relation-ml/artifacts/phase2/records_features.pkl \
  --candidates /content/drive/MyDrive/entity-relation-ml/artifacts/phase3/candidate_pairs.tsv \
  --ground-truth /content/drive/MyDrive/entity-relation-ml/dataset/train/train_ground_truth.tsv \
  --model-dir /content/drive/MyDrive/entity-relation-ml/models \
  --neg-ratio 20 \
  --n-jobs 4
```

Phase 5 currently expects one combined records pickle. For the full chunked workflow, complete the Phase 4/5 chunk-reader migration before using this command on the complete dataset.

## 4. Push only code

From the project root on your local machine:

```powershell
git add .gitignore requirements.txt PHASE_WORKFLOW.md COLAB.md src utils
git commit -m "Prepare phased entity matching workflow for Colab"
git push -u origin main
```

The ignore rules exclude datasets, logs, generated artifacts, candidate tables, models, virtual environments, and Python caches.
