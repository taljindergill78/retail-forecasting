"""
Promote a RetailSalesForecaster model version to a given MLflow stage (Staging or Production).

Use this after you've run Train (and optionally Evaluate) and want to mark a specific
run's model as Staging (for review) or Production (for serving).

Run from project root:

  # Promote the model version that came from a specific run to Production
  python scripts/mlflow_promote_model.py --run-id <RUN_ID> --stage Production

  # Or promote by version number (from MLflow UI → Models → RetailSalesForecaster)
  python scripts/mlflow_promote_model.py --version <VERSION> --stage Production

  # Use Staging first if you want to review before Production
  python scripts/mlflow_promote_model.py --run-id <RUN_ID> --stage Staging

Requires:
  - MLflow tracking server reachable (set MLFLOW_TRACKING_URI or use infra/sagemaker_config.env).
  - The run must have registered a model under the name RetailSalesForecaster (train.py does this).
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import getpass
from typing import Optional
import logging
import warnings

# Load env from infra/sagemaker_config.env if present (for MLFLOW_TRACKING_URI)
_project_root = Path(__file__).resolve().parent.parent
_env_file = _project_root / "infra" / "sagemaker_config.env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file)

import mlflow
from mlflow.tracking import MlflowClient

# Reduce noisy MLflow warnings in this helper
logging.getLogger("mlflow.tracking.request_header.registry").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="``mlflow.tracking.client.MlflowClient.transition_model_version_stage`` is deprecated",
)

MODEL_NAME = "RetailSalesForecaster"
VALID_STAGES = ("Staging", "Production", "Archived", "None")


def get_tracking_uri():
    """Use same logic as rest of project: env var (e.g. from infra/sagemaker_config.env) or local SQLite."""
    import os
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    return f"sqlite:///{_project_root / 'mlflow.db'}"


def promote_by_run_id(client: MlflowClient, run_id: str, stage: str, reason: Optional[str] = None) -> None:
    """Find the model version linked to this run_id and transition it to the given stage."""
    versions = client.search_model_versions(f"name = '{MODEL_NAME}' and run_id = '{run_id}'")
    if not versions:
        raise ValueError(
            f"No registered model version found for run_id={run_id}. "
            f"Ensure the run logged a model to '{MODEL_NAME}' (train.py does this)."
        )
    version = versions[0]
    _transition(client, version.version, stage)
    _add_promotion_tags(
        client=client,
        version=str(version.version),
        stage=stage,
        run_id=run_id,
        reason=reason,
    )


def promote_by_version(client: MlflowClient, version: str, stage: str, reason: Optional[str] = None) -> None:
    """Transition the given model version (e.g. '3') to the given stage."""
    mv = client.get_model_version(name=MODEL_NAME, version=version)
    _transition(client, version, stage)
    _add_promotion_tags(
        client=client,
        version=str(version),
        stage=stage,
        run_id=getattr(mv, "run_id", None),
        reason=reason,
    )


def _transition(client: MlflowClient, version: str, stage: str) -> None:
    # When promoting to Production, we usually want only one Production version.
    # archive_existing_versions=True moves any current Production version to Archived.
    archive_existing = (stage == "Production")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    print(f"✅ Promoted {MODEL_NAME} version {version} to stage '{stage}'.")
    if archive_existing:
        print("   (Previous Production version, if any, was moved to Archived.)")


def _add_promotion_tags(
    client: MlflowClient,
    version: str,
    stage: str,
    run_id: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    """Attach basic promotion metadata as tags on the model version (and run, if known)."""
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    user = getpass.getuser()

    # Model version tags
    client.set_model_version_tag(MODEL_NAME, version, "promotion_stage", stage)
    client.set_model_version_tag(MODEL_NAME, version, "promotion_by", user)
    client.set_model_version_tag(MODEL_NAME, version, "promotion_time_utc", timestamp)
    if reason:
        client.set_model_version_tag(MODEL_NAME, version, "promotion_reason", reason)

    # Optional: mirror some tags on the run itself
    if run_id:
        try:
            client.set_tag(run_id, "last_promotion_stage", stage)
            client.set_tag(run_id, "last_promotion_time_utc", timestamp)
            if reason:
                client.set_tag(run_id, "last_promotion_reason", reason)
        except Exception:
            # Tagging the run is best-effort; do not fail promotion if this breaks.
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Promote a RetailSalesForecaster model version to Staging or Production."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", type=str, help="MLflow run ID that produced the model (e.g. from mlflow_run_id.txt)")
    group.add_argument("--version", type=str, help="Model version number (e.g. 2 or 3) from MLflow UI")
    parser.add_argument(
        "--stage",
        type=str,
        default="Production",
        choices=VALID_STAGES,
        help="Target stage (default: Production)",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default=None,
        help="Optional free-text reason for promotion (stored as MLflow tags).",
    )
    args = parser.parse_args()

    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    print(f"📊 MLflow tracking URI: {uri}")

    client = MlflowClient()

    try:
        if args.run_id:
            promote_by_run_id(client, args.run_id, args.stage, args.reason)
        else:
            promote_by_version(client, args.version, args.stage, args.reason)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
