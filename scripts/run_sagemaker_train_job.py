"""
Launch the Train SageMaker Processing job (reads features + baselines from S3, trains models, logs to MLflow).

Run from project root:

    python scripts/run_sagemaker_train_job.py

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
  - S3_FEATURES_PREFIX (default s3://{S3_BUCKET}/sagemaker/features)
  - S3_BASELINES_PREFIX (default s3://{S3_BUCKET}/sagemaker/baselines)
  - S3_TRAIN_OUTPUT_PREFIX (default s3://{S3_BUCKET}/sagemaker/models)
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

S3_FEATURES_PREFIX = os.environ.get("S3_FEATURES_PREFIX") or f"s3://{S3_BUCKET}/sagemaker/features"
S3_BASELINES_PREFIX = os.environ.get("S3_BASELINES_PREFIX") or f"s3://{S3_BUCKET}/sagemaker/baselines"
S3_TRAIN_OUTPUT_PREFIX = os.environ.get("S3_TRAIN_OUTPUT_PREFIX") or f"s3://{S3_BUCKET}/sagemaker/models"

FEATURES_PATH = "/opt/ml/processing/input/features"
DATA_PATH = "/opt/ml/processing/input/data"
OUTPUT_PATH = "/opt/ml/processing/output"

env = {
    "SPLITS_DIR": FEATURES_PATH,
    "DATA_DIR": DATA_PATH,
    "MODELS_DIR": OUTPUT_PATH,
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
            source=S3_FEATURES_PREFIX,
            destination=FEATURES_PATH,
            input_name="features",
        ),
        ProcessingInput(
            source=S3_BASELINES_PREFIX,
            destination=DATA_PATH,
            input_name="data",
        ),
    ],
    outputs=[
        ProcessingOutput(
            source=OUTPUT_PATH,
            destination=S3_TRAIN_OUTPUT_PREFIX,
            output_name="models",
        )
    ],
    arguments=["-m", "src.model.train"],
    job_name=f"retail-train-job-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
)

print("Train job submitted. Check SageMaker console, S3 sagemaker/models/, and MLflow UI.")

