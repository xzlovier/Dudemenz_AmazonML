# Business Entity Resolution

This repository contains the pipeline for the Amazon ML Challenge - Business Entity Resolution.

## Structure
- `src/`: Contains the pipeline scripts.
  - `step1_setup.py`: Data loading and validation.
  - `step2_eda_fast.py`: Exploratory Data Analysis.
  - `step3_harness.py`: Local evaluation metrics (F0.5, recall, reduction ratio).
  - `step4_normalize.py`: Normalizes names and addresses, extracts PIN/ZIP.
  - `step5_blocking.py`: Generates candidate pairs using TF-IDF + FAISS, MinHash, and Metaphone.
  - `step6_shrink.py`: Shrinks candidate pairs using string similarity.
- `requirements.txt`: Environment dependencies.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure the `dataset/` folder is placed in the root of the project (at the same level as `code/` and `output/`).
3. Run normalization:
   ```bash
   python code/business_entity_resolution/src/step4_normalize.py
   ```
4. Run blocking:
   ```bash
   python code/business_entity_resolution/src/step5_blocking.py
   ```
5. Run shrinking (which outputs the final `candidate_pairs.tsv` to `output/`):
   ```bash
   python code/business_entity_resolution/src/step6_shrink.py
   ```
