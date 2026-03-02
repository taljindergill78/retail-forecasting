# Scripts for launching SageMaker jobs

These scripts submit SageMaker Processing jobs from your laptop (they call the AWS API; the jobs run in the cloud).

**Do not put secrets in the script files.** All config (role ARN, ECR URI, bucket, subnets, DB host, RDS secret, etc.) is read from **environment variables**. You can:

1. **Use a gitignored env file:** Copy `infra/sagemaker_config.env.sample` to `infra/sagemaker_config.env`, fill in your values, and run the script. The script loads `infra/sagemaker_config.env` automatically if it exists.
2. **Export in the shell:** `export SAGEMAKER_ROLE_ARN=... ECR_IMAGE_URI=...` etc., then run the script.

The script will error with a clear message if a required variable is missing. Required vars are listed in the script docstring and in the sample file.

**Scripts are committed to git**; only the env file with real values is gitignored.
