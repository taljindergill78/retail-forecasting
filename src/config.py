"""
Load pipeline parameters from params.yaml at project root.

All pipeline scripts (split, feature_eng, train, evaluate_models, run_baselines)
should use load_params() so that config is single source of truth and DVC
param tracking matches actual behavior.

For SageMaker (Phase B): Path and MLflow settings can be overridden via environment
variables so the same code works locally (DVC) and on SageMaker. See get_*_dir() and
get_mlflow_tracking_uri() below.
"""
import os
from pathlib import Path

import yaml


def get_project_root():
    """Project root = directory containing src/, params.yaml, dvc.yaml."""
    # When running as python src/data/split.py or python src/features/feature_eng.py, cwd is project root
    if Path.cwd().joinpath("params.yaml").exists():
        return Path.cwd()
    # When running from elsewhere, resolve from this file: src/config.py -> parent.parent = root
    return Path(__file__).resolve().parent.parent


def load_params(params_path=None):
    """
    Load params.yaml from project root.

    Returns:
        dict: Full params structure (split, features, models, evaluation, manifests).
    """
    root = get_project_root()
    path = params_path or root / "params.yaml"
    if not path.exists():
        raise FileNotFoundError(f"params.yaml not found at {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Path and MLflow overrides for SageMaker (Phase B)
# If env var is set, use it (SageMaker); otherwise use project-root-relative default (local).
# See docs/PHASE_B_STEP_6_ADAPT_SCRIPTS_FOR_SAGEMAKER_GUIDE.md
# ---------------------------------------------------------------------------

def get_data_dir():
    """Directory for pipeline data (splits, features, model outputs, etc.). Default: project_root/data."""
    root = get_project_root()
    path = os.environ.get("DATA_DIR")
    return Path(path) if path else root / "data"


def get_splits_dir():
    """Directory for train/val/test splits and feature CSVs. Default: data_dir/splits."""
    data = get_data_dir()
    path = os.environ.get("SPLITS_DIR")
    return Path(path) if path else data / "splits"


def get_models_dir():
    """Directory for saved models. Default: project_root/models."""
    root = get_project_root()
    path = os.environ.get("MODELS_DIR")
    return Path(path) if path else root / "models"


def get_reports_dir():
    """Directory for reports and figures. Default: project_root/reports."""
    root = get_project_root()
    path = os.environ.get("REPORTS_DIR")
    return Path(path) if path else root / "reports"


def get_mlflow_tracking_uri():
    """
    MLflow tracking server URI. If MLFLOW_TRACKING_URI is set (e.g. on SageMaker),
    use it; otherwise default to local SQLite (local runs).
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    root = get_project_root()
    return f"sqlite:///{root / 'mlflow.db'}"
