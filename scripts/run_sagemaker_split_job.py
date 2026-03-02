"""
Launch the Split SageMaker Processing job (reads from RDS, writes splits to S3).

Run from project root:  python scripts/run_sagemaker_split_job.py

Config: Set environment variables (e.g. in a .env file or shell). Do NOT commit
secrets. Use infra/sagemaker_config.env (gitignored) or export in the shell.
Required: SAGEMAKER_ROLE_ARN, ECR_IMAGE_URI, S3_BUCKET, DB_HOST, RDS_SECRET_NAME,
  SUBNET_IDS, SECURITY_GROUP_IDS (comma-separated).
Optional: AWS_REGION (default us-west-2), DB_NAME (default retailds),
  SAGEMAKER_INSTANCE_TYPE (default ml.t3.medium; use ml.m5.xlarge if you have quota).
"""
import os
import re
from datetime import datetime
from pathlib import Path

# Load from env file if present (do not commit that file with real values)
_env_file = Path(__file__).resolve().parent.parent / "infra" / "sagemaker_config.env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file)

import sagemaker
from sagemaker.network import NetworkConfig
from sagemaker.processing import Processor, ProcessingOutput

# --- 1) Config from environment (no defaults for secrets) ---
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN") or os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN")
ECR_IMAGE_URI = os.environ.get("ECR_IMAGE_URI")
S3_BUCKET = os.environ.get("S3_BUCKET")
_raw_subnets = os.environ.get("SUBNET_IDS", "")
SUBNET_IDS = [s.strip() for s in _raw_subnets.split(",") if s.strip()] if _raw_subnets else []
_raw_sgs = os.environ.get("SECURITY_GROUP_IDS") or os.environ.get("SAGEMAKER_SECURITY_GROUP_ID", "")
SECURITY_GROUP_IDS = [s.strip() for s in _raw_sgs.split(",") if s.strip()] if _raw_sgs else []
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME", "retailds")
RDS_SECRET_NAME = os.environ.get("RDS_SECRET_NAME")
# Instance type: default ml.t3.medium (often has quota); set SAGEMAKER_INSTANCE_TYPE for larger (e.g. ml.m5.xlarge)
INSTANCE_TYPE = os.environ.get("SAGEMAKER_INSTANCE_TYPE", "ml.t3.medium")

for name, val in [
    ("SAGEMAKER_ROLE_ARN", SAGEMAKER_ROLE_ARN),
    ("ECR_IMAGE_URI", ECR_IMAGE_URI),
    ("S3_BUCKET", S3_BUCKET),
    ("DB_HOST", DB_HOST),
    ("RDS_SECRET_NAME", RDS_SECRET_NAME),
]:
    if not val:
        raise SystemExit(f"Missing required env var: {name}. Set it in the shell or in infra/sagemaker_config.env (gitignored).")

# AWS requires role ARN with a real 12-digit account ID (no placeholder)
if "ACCOUNT_ID" in (SAGEMAKER_ROLE_ARN or ""):
    raise SystemExit(
        "SAGEMAKER_ROLE_ARN must use your real AWS account ID, not the placeholder 'ACCOUNT_ID'. "
        "In infra/sagemaker_config.env set e.g. SAGEMAKER_EXECUTION_ROLE_ARN=arn:aws:iam::123456789012:role/retail-sagemaker-execution-role "
        "(replace 123456789012 with your 12-digit account ID from AWS Console → account dropdown)."
    )
if not re.match(r"arn:aws[a-z\-]*:iam::\d{12}:role/", SAGEMAKER_ROLE_ARN or ""):
    raise SystemExit(
        "SAGEMAKER_ROLE_ARN must match arn:aws:iam::<12-digit-account-id>:role/<role-name>. "
        "Check infra/sagemaker_config.env (or SAGEMAKER_EXECUTION_ROLE_ARN)."
    )

if not SUBNET_IDS or not SECURITY_GROUP_IDS:
    raise SystemExit("SUBNET_IDS and SECURITY_GROUP_IDS (or SAGEMAKER_SECURITY_GROUP_ID) are required for the job to reach RDS. Use comma-separated values.")

SPLITS_OUTPUT_PATH = "/opt/ml/processing/output"
S3_SPLITS_PREFIX = f"s3://{S3_BUCKET}/sagemaker/splits"

env = {
    "SPLITS_DIR": SPLITS_OUTPUT_PATH,
    "DB_HOST": DB_HOST,
    "DB_NAME": DB_NAME,
    "RDS_SECRET_NAME": RDS_SECRET_NAME,
    "AWS_REGION": AWS_REGION,
}

# --- 2) Create Processor and run (SageMaker SDK 2.x uses network_config) ---
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
    inputs=[],
    outputs=[
        ProcessingOutput(
            source=SPLITS_OUTPUT_PATH,
            destination=S3_SPLITS_PREFIX,
            output_name="splits",
        )
    ],
    arguments=["-m", "src.data.split"],
    job_name=f"retail-split-job-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
)

print("Job submitted. Check SageMaker console → Processing → Processing jobs.")
