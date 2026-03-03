"""
Launch the Evaluate SageMaker Processing job (reads train outputs + features from S3, writes reports to S3).

Run from project root:

    python scripts/run_sagemaker_evaluate_job.py

Config: Set environment variables (e.g. in a .env file or shell). Do NOT commit
secrets. Use infra/sagemaker_config.env (gitignored) or export in the shell.

Required:
  - SAGEMAKER_ROLE_ARN or SAGEMAKER_EXECUTION_ROLE_ARN
  - ECR_IMAGE_URI
  - S3_BUCKET
  - MLFLOW_TRACKING_URI
  - SUBNET_IDS, SECURITY_GROUP_IDS (comma-separated; or SAGEMAKER_SECURITY_GROUP_ID)

Optional:
  - AWS_REGION (default us-west-2)
  - SAGEMAKER_INSTANCE_TYPE (default ml.t3.medium; use ml.m5.xlarge if you have quota)
  - S3_TRAIN_OUTPUT_PREFIX (default s3://{S3_BUCKET}/sagemaker/models)
  - S3_FEATURES_PREFIX (default s3://{S3_BUCKET}/sagemaker/features)
  - S3_REPORTS_OUTPUT_PREFIX (default s3://{S3_BUCKET}/sagemaker/output)
"""

import os
import re
from datetime import datetime
from pathlib import Path

_env_file = Path(__file__).resolve().parent.parent / "infra" / "sagemaker_config.env"
if _env_file.exists():
    from dotenv import load_dotenv

    load_dotenv(_env_file)

import sagemaker
from sagemaker.network import NetworkConfig
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor

AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN") or os.environ.get(
    "SAGEMAKER_EXECUTION_ROLE_ARN"
)
ECR_IMAGE_URI = os.environ.get("ECR_IMAGE_URI")
S3_BUCKET = os.environ.get("S3_BUCKET")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
_raw_subnets = os.environ.get("SUBNET_IDS", "")
SUBNET_IDS = [s.strip() for s in _raw_subnets.split(",") if s.strip()] if _raw_subnets else []
_raw_sgs = os.environ.get("SECURITY_GROUP_IDS") or os.environ.get(
    "SAGEMAKER_SECURITY_GROUP_ID", ""
)
SECURITY_GROUP_IDS = [s.strip() for s in _raw_sgs.split(",") if s.strip()] if _raw_sgs else []
INSTANCE_TYPE = os.environ.get("SAGEMAKER_INSTANCE_TYPE", "ml.t3.medium")

for name, val in [
    ("SAGEMAKER_ROLE_ARN", SAGEMAKER_ROLE_ARN),
    ("ECR_IMAGE_URI", ECR_IMAGE_URI),
    ("S3_BUCKET", S3_BUCKET),
    ("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI),
]:
    if not val:
        raise SystemExit(
            f"Missing required env var: {name}. Set it in the shell or in infra/sagemaker_config.env (gitignored)."
        )

if "ACCOUNT_ID" in (SAGEMAKER_ROLE_ARN or ""):
    raise SystemExit(
        "SAGEMAKER_ROLE_ARN must use your real AWS account ID, not the placeholder 'ACCOUNT_ID'. "
        "In infra/sagemaker_config.env set e.g. "
        "SAGEMAKER_EXECUTION_ROLE_ARN=arn:aws:iam::123456789012:role/retail-sagemaker-execution-role "
        "(replace 123456789012 with your 12-digit account ID from AWS Console → account dropdown)."
    )
if not re.match(r"arn:aws[a-z\-]*:iam::\d{12}:role/", SAGEMAKER_ROLE_ARN or ""):
    raise SystemExit(
        "SAGEMAKER_ROLE_ARN must match arn:aws:iam::<12-digit-account-id>:role/<role-name>. "
        "Check infra/sagemaker_config.env (or SAGEMAKER_EXECUTION_ROLE_ARN)."
    )

if not SUBNET_IDS or not SECURITY_GROUP_IDS:
    raise SystemExit(
        "SUBNET_IDS and SECURITY_GROUP_IDS (or SAGEMAKER_SECURITY_GROUP_ID) are required for the job. "
        "Use comma-separated values."
    )

S3_TRAIN_OUTPUT_PREFIX = os.environ.get("S3_TRAIN_OUTPUT_PREFIX") or f"s3://{S3_BUCKET}/sagemaker/models"
S3_FEATURES_PREFIX = os.environ.get("S3_FEATURES_PREFIX") or f"s3://{S3_BUCKET}/sagemaker/features"
S3_REPORTS_OUTPUT_PREFIX = os.environ.get("S3_REPORTS_OUTPUT_PREFIX") or f"s3://{S3_BUCKET}/sagemaker/output"

DATA_INPUT_PATH = "/opt/ml/processing/input/data"
SPLITS_INPUT_PATH = "/opt/ml/processing/input/features"
REPORTS_OUTPUT_BASE = "/opt/ml/processing/output"
REPORTS_DIR_PATH = "/opt/ml/processing/output/reports"
FIGURES_DIR_PATH = "/opt/ml/processing/output/figures"

env = {
    "DATA_DIR": DATA_INPUT_PATH,
    "SPLITS_DIR": SPLITS_INPUT_PATH,
    "REPORTS_DIR": REPORTS_DIR_PATH,   # CSV will go under /output/reports
    "FIGURES_DIR": FIGURES_DIR_PATH,   # figures will go under /output/figures
    "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    "AWS_REGION": AWS_REGION,
}

network_config = NetworkConfig(
    subnets=SUBNET_IDS,
    security_group_ids=SECURITY_GROUP_IDS,
)

processor = Processor(
    image_uri=ECR_IMAGE_URI,
    role=SAGEMAKER_ROLE_ARN,
    instance_count=1,
    instance_type=INSTANCE_TYPE,
    volume_size_in_gb=30,
    network_config=network_config,
    env=env,
    sagemaker_session=sagemaker.Session(boto_session=None),
)

processor.run(
    inputs=[
        ProcessingInput(
            source=S3_TRAIN_OUTPUT_PREFIX,
            destination=DATA_INPUT_PATH,
            input_name="data",
        ),
        ProcessingInput(
            source=S3_FEATURES_PREFIX,
            destination=SPLITS_INPUT_PATH,
            input_name="features",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source=REPORTS_OUTPUT_BASE,
            destination=S3_REPORTS_OUTPUT_PREFIX,
            output_name="reports",
        )
    ],
    arguments=["-m", "src.model.evaluate_models"],
    job_name=f"retail-evaluate-job-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
)

print("Evaluate job submitted. Check SageMaker console, S3 reports/output prefix, and MLflow UI.")

