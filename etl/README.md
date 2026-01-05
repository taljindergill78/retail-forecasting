# ETL Scripts

This folder contains all Extract, Transform, Load (ETL) scripts for the project.

## Structure

- `glue/` - AWS Glue ETL scripts
  - These scripts run in AWS Glue to move data from S3 → RDS Postgres
  - They are version-controlled here for reproducibility and documentation

## Current ETL Flow

1. **Source**: Raw CSV files in `s3://retail-ml-taljinder-2025/raw/walmart/`
2. **Process**: AWS Glue job (`walmart-s3-to-rds-etl`)
3. **Destination**: RDS Postgres (`retailds` schema)
   - `dim_store` (45 stores)
   - `dim_dept` (81 departments)
   - `fact_sales_weekly` (421,570 sales records)
   - `fact_sales_weekly_stg` (staging table)

## How to Use

1. Copy the Glue script from AWS Glue console
2. Save it in `glue/` folder with a descriptive name (e.g., `walmart_s3_to_rds.py`)
3. This script can be:
   - Referenced when creating/updating Glue jobs
   - Used for documentation and code reviews
   - Version-controlled alongside schema changes

## Notes

- The ETL script implements idempotent loading (safe to rerun)
- Uses staging table pattern for upserts
- Handles data normalization and type casting

