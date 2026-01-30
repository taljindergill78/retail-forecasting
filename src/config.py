"""
Load pipeline parameters from params.yaml at project root.

All pipeline scripts (split, feature_eng, train, evaluate_models, run_baselines)
should use load_params() so that config is single source of truth and DVC
param tracking matches actual behavior.
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
