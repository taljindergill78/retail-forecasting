# Retail Sales Forecasting (Walmart)

A production-style ML pipeline for weekly retail sales forecasting. The project covers data ingestion (S3 → RDS), reproducible ML with DVC, experiment tracking with MLflow, and a path toward deployment on AWS (SageMaker, then serving and EKS as later phases).

**Status:** The full pipeline runs locally (split → EDA → features → baselines → train → evaluate). AWS Phase B (SageMaker) is in progress: infrastructure and script adaptation are done; running and chaining SageMaker jobs is the next step.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [High-Level Architecture](#high-level-architecture)
- [What's Implemented](#whats-implemented)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup and Run (Local)](#setup-and-run-local)
- [Pipeline Stages (DVC)](#pipeline-stages-dvc)
- [Configuration](#configuration)
- [AWS and Phase B (SageMaker)](#aws-and-phase-b-sagemaker)
- [Next Steps (Planned)](#next-steps-planned)
- [References](#references)

---

## What This Project Does

1. **Data:** Raw Walmart-style data (train, features, stores CSVs) lands in S3. AWS Glue ETL loads it into RDS PostgreSQL (schema `retailds`). An analytics view `vw_sales_analytics` joins fact and dimension tables for ML.
2. **Split:** The pipeline reads from RDS (`retailds.vw_sales_analytics`), splits by date into train/validation/test, and writes CSVs. No shuffling—chronological order is preserved for time-series.
3. **EDA:** Exploratory analysis on the splits: schema checks, time-series plots, segment analysis, dashboard-ready tables and figures under `reports/eda/`.
4. **Features:** Time, lag, rolling, holiday-window, and markdown features are built from the splits. All behavior is driven by `params.yaml`.
5. **Baselines:** Naive, seasonal naive, moving average, seasonal average, and hybrid baselines are evaluated on validation data with WMAE and segmented metrics. Results are saved so the train stage can compare against them.
6. **Train:** Ridge, LightGBM, CatBoost, and XGBoost are tuned on validation data; a weighted ensemble is built. The best model (must beat the best baseline on both RMSE and WMAE) is selected, retrained on train+val, and evaluated on the test set. All runs are logged to MLflow; the best model is registered. Test predictions and comparison CSVs are written to `data/`.
7. **Evaluate:** Loads train outputs (model comparison, test predictions, MLflow run ID), computes test-set metrics and segmented evaluation, writes an evaluation report and figures to `reports/`.

The same codebase is prepared to run on **AWS SageMaker** (Phase B): paths and MLflow URI are configurable via environment variables so jobs can read/write S3-mounted paths and log to a shared MLflow server.

---

## High-Level Architecture

```
S3 (raw CSVs) → Glue ETL → RDS (retailds: fact_sales_weekly, dim_store, dim_dept, vw_sales_analytics)
                                    ↓
                    DVC pipeline (local or SageMaker):
                    split → eda → features → baselines → train → evaluate
                                    ↓
                    data/splits, data/*.csv, models/, reports/
                    MLflow (local SQLite or remote server)
```

- **Local:** You run `dvc repro` from the project root. Scripts read/write under `data/`, `models/`, `reports/` and use `params.yaml` and (optionally) `.env` for DB credentials.
- **SageMaker (Phase B):** The same scripts run inside containers. Inputs/outputs are wired via env vars (`SPLITS_DIR`, `DATA_DIR`, `MODELS_DIR`, `REPORTS_DIR`, `MLFLOW_TRACKING_URI`) so they use SageMaker Processing/Training job paths; outputs are uploaded to S3.

---

## What's Implemented

| Area | Implemented | Notes |
|------|-------------|--------|
| **Data & ETL** | ✅ | S3 raw data; Glue script `etl/glue/walmart_s3_to_rds.py`; RDS schema and view `sql/01_create_schema.sql`, `sql/02_create_analytics_view.sql`; Lambdas for cleanup/upsert/validation in `infra/lambdas/` |
| **Manifests** | ✅ | `tools/run_manifests.py` / `make_manifest.py`; manifest JSONs fingerprint S3 URIs for pipeline invalidation |
| **Split** | ✅ | `src/data/split.py` — reads RDS, chronological split, writes train/val/test CSVs; supports Secrets Manager or `.env` for DB credentials |
| **EDA** | ✅ | `src/eda/eda.py` — schema checks, time-series and segment plots, dashboard tables in `reports/eda/figures` and `reports/eda/tables` |
| **Features** | ✅ | `src/features/feature_eng.py` — time, lags, rolling, holiday, markdown; config from `params.yaml` |
| **Baselines** | ✅ | `src/model/run_baselines.py` — multiple baselines, WMAE, segmented evaluation; writes `data/baseline_results.csv` and segment CSVs |
| **Train** | ✅ | `src/model/train.py` — Ridge, LightGBM, CatBoost, XGBoost + weighted ensemble; MLflow logging and model registration; test predictions and comparison CSVs |
| **Evaluate** | ✅ | `src/model/evaluate_models.py` — test metrics, report CSV, figures; can attach artifacts to the same MLflow run as train |
| **Config** | ✅ | `src/config.py` — `load_params()`, `get_data_dir()`, `get_splits_dir()`, `get_models_dir()`, `get_reports_dir()`, `get_mlflow_tracking_uri()` for local and SageMaker |
| **DVC** | ✅ | Full pipeline in `dvc.yaml`; remote S3 for cache (`dvc push` / `dvc pull`) |
| **Docker** | ✅ | `Dockerfile` for SageMaker (Python 3.11, `WORKDIR /opt/ml/code`, `ENTRYPOINT ["python"]`) |
| **Phase B (SageMaker)** | In progress | Steps 1–6 done (IAM, VPC, S3, MLflow on EC2, image in ECR, scripts adapted for env-based paths). Step 7 (run SageMaker jobs one by one, then chain) is next. |

Nothing in this table is overstated: if a feature is listed as implemented, it exists and is used in the current pipeline or infrastructure.

---

## Project Structure

```
retail-forecasting/
├── README.md                 # This file
├── params.yaml               # Pipeline parameters (split dates, feature config, model grids, evaluation)
├── dvc.yaml                  # DVC pipeline definition
├── dvc.lock                  # Locked versions of deps/params/outs
├── requirements.txt          # Python dependencies
├── Dockerfile                # Image for SageMaker (and future serving)
├── .dvc/                     # DVC config (e.g. remote = s3)
│
├── src/
│   ├── config.py             # load_params(), get_*_dir(), get_mlflow_tracking_uri()
│   ├── data/
│   │   └── split.py          # RDS → train/val/test CSVs
│   ├── eda/
│   │   └── eda.py            # EDA plots and tables
│   ├── features/
│   │   └── feature_eng.py     # Feature engineering
│   └── model/
│       ├── baseline.py       # Baseline forecast functions
│       ├── evaluate.py       # WMAE and segment evaluation (library)
│       ├── run_baselines.py  # Baseline comparison script
│       ├── train.py          # Model training and selection
│       ├── evaluate_models.py # Test evaluation and report
│       ├── generate_all_test_predictions.py  # Optional: regenerate all test preds
│       └── inspect_predictions.py            # Optional: inspect predictions by store/dept
│
├── tools/
│   ├── run_manifests.py      # Generate manifest JSONs from params
│   └── make_manifest.py     # Manifest building helpers
│
├── sql/
│   ├── 01_create_schema.sql  # Tables: dim_store, dim_dept, fact_sales_weekly, staging
│   ├── 02_create_analytics_view.sql  # vw_sales_analytics
│   └── README.md
│
├── etl/
│   ├── glue/                 # AWS Glue ETL script (S3 → RDS)
│   └── README.md
│
├── infra/
│   └── lambdas/              # Pre-ETL cleanup, post-ETL upsert, post-ETL validation
│
├── data/                     # DVC-tracked (or local) data outputs
│   ├── splits/               # train.csv, val.csv, test.csv, *_features.csv
│   ├── baseline_results.csv, baseline_segments_*, model_comparison.csv, etc.
│   └── test_predictions_all/
│
├── models/                   # Best model artifacts (e.g. best_model_meta.pkl, best_model.json)
├── reports/                  # EDA and evaluation outputs
│   ├── eda/figures, eda/tables
│   ├── figures/              # Evaluation figures
│   └── model_evaluation_report.csv
│
├── data_manifests/           # Manifest JSONs (DVC outs)
│
├── docs/                     # Phase B step-by-step guides (e.g. adapt scripts, run SageMaker jobs)
└── reports/                  # Planning and reference docs (e.g. NEXT_STEPS_AWS_SAGEMAKER_PHASE_B.md)
```

---

## Prerequisites

- **Python 3.10+** (3.11 used in Docker)
- **DVC** (with S3 remote configured if you use `dvc push`/`dvc pull`)
- **Access to RDS** for the split stage: PostgreSQL database `retailds` with schema and view created from `sql/`. Credentials via `.env` (e.g. `DB_HOST`, `DB_USER`, `DB_PASSWORD`) or AWS Secrets Manager (`RDS_SECRET_NAME`, `AWS_REGION`)

See `requirements.txt` for Python packages (pandas, scikit-learn, LightGBM, CatBoost, XGBoost, MLflow, psycopg2-binary, boto3, etc.).

---

## Setup and Run (Local)

1. **Clone and install**
   ```bash
   cd retail-forecasting
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   pip install dvc  # if not already installed
   ```

2. **Configure DVC remote** (if not already set)
   ```bash
   dvc remote add s3remote s3://<your-bucket>/dvc   # or use existing .dvc/config
   ```

3. **Database**
   - Ensure RDS has the schema and view: run `sql/01_create_schema.sql` and `sql/02_create_analytics_view.sql` in the `retailds` database.
   - Create a `.env` in the project root (do not commit) with at least:
     - `DB_HOST`, `DB_NAME` (e.g. `retailds`), `DB_USER`, `DB_PASSWORD`
     - Or use `RDS_SECRET_NAME` and `AWS_REGION` for Secrets Manager.

4. **Reproduce the pipeline**
   ```bash
   dvc repro
   ```
   This runs, in order: manifests → split → eda → features → baselines → train → evaluate. Outputs go to `data/`, `models/`, and `reports/`.

5. **Single stage**
   ```bash
   dvc repro evaluate   # or split, eda, features, baselines, train
   ```

6. **MLflow UI** (optional)
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
   Open the URL shown to browse experiments and the registered model.

---

## Pipeline Stages (DVC)

| Stage | Command | Main outputs |
|-------|--------|---------------|
| `manifests` | `python tools/run_manifests.py` | `data_manifests/walmart/*.manifest.json` |
| `split` | `python src/data/split.py` | `data/splits/train.csv`, `val.csv`, `test.csv` |
| `eda` | `python src/eda/eda.py` | `reports/eda/figures/`, `reports/eda/tables/` |
| `features` | `python src/features/feature_eng.py` | `data/splits/train_features.csv`, `val_features.csv`, `test_features.csv` |
| `baselines` | `python src/model/run_baselines.py` | `data/baseline_results.csv`, `data/baseline_segments_all.csv` |
| `train` | `python src/model/train.py` | `models/`, `data/model_comparison.csv`, `data/final_test_results.json`, `data/test_predictions_all/`, `data/mlflow_run_id.txt` |
| `evaluate` | `python src/model/evaluate_models.py` | `reports/model_evaluation_report.csv`, `reports/figures/` |

All paths used by these scripts are configurable via `src/config.py` (and env vars on SageMaker). See `dvc.yaml` for exact dependencies and parameters.

---

## Configuration

- **`params.yaml`:** Single source of truth for split dates, feature options (lags, rolling windows, negative-sales strategy), model hyperparameter grids, and evaluation (e.g. holiday weight, plot samples). DVC tracks param changes and triggers repro when needed.
- **`src/config.py`:** Loads `params.yaml` and exposes directory helpers that respect environment variables (`DATA_DIR`, `SPLITS_DIR`, `MODELS_DIR`, `REPORTS_DIR`, `MLFLOW_TRACKING_URI`). Local runs use project-relative defaults; SageMaker jobs set these to container paths and the MLflow server URL.

---

## AWS and Phase B (SageMaker)

The pipeline is adapted so the **same code** runs locally and on SageMaker:

- **Done:** IAM role for SageMaker, VPC/security groups so SageMaker can reach RDS and MLflow, S3 layout for pipeline outputs, MLflow server on EC2 (RDS backend + S3 artifacts), Docker image in ECR, and script changes so all I/O and MLflow URI come from `src/config.py` (env-driven). See `docs/PHASE_B_STEP_6_ADAPT_SCRIPTS_FOR_SAGEMAKER_GUIDE.md` for what was changed and why.
- **Next:** Run SageMaker jobs one at a time (split → features → train → evaluate), then chain them into a pipeline and add a model registration step. The runbook and example job definitions are in `docs/PHASE_B_STEP_7_RUN_SAGEMAKER_JOBS_GUIDE.md`.

High-level planning and rationale (e.g. MLflow vs SageMaker Model Registry, approval workflow) are in `reports/2. NEXT_STEPS_AWS_SAGEMAKER_PHASE_B.md` and related reports.

---

## Next Steps (Planned)

These are intended directions, not promises; they will be updated as work is done.

- **Phase B (SageMaker):** Run and verify each SageMaker job (split, features, train, evaluate), then define a SageMaker Pipeline and a register-model step. Optionally run EDA or baselines as SageMaker jobs.
- **Model registry and approval:** Use SageMaker Model Registry with a PendingManualApproval → Approved flow so only approved models are used downstream.
- **Serving (Phase C):** FastAPI (or similar) in Docker, loading only approved models from the registry.
- **Deployment (Phase D):** Deploy the serving API on EKS.
- **CI/CD (Phase E):** Automate tests and deployment when new models are approved.
- **Monitoring and batch inference:** Add monitoring, alarms, and optional batch inference (later phases).

---

## References

- **Pipeline and config:** `dvc.yaml`, `params.yaml`, `src/config.py`
- **Database:** `sql/README.md`, `sql/01_create_schema.sql`, `sql/02_create_analytics_view.sql`
- **ETL:** `etl/README.md`, `etl/glue/walmart_s3_to_rds.py`
- **Phase B (SageMaker):** `docs/PHASE_B_STEP_6_ADAPT_SCRIPTS_FOR_SAGEMAKER_GUIDE.md`, `docs/PHASE_B_STEP_7_RUN_SAGEMAKER_JOBS_GUIDE.md`, `reports/2. NEXT_STEPS_AWS_SAGEMAKER_PHASE_B.md`
