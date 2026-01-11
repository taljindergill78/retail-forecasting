import os
import json
import boto3
import psycopg2

def get_secret(secret_id: str) -> dict:
    sm = boto3.client("secretsmanager")
    resp = sm.get_secret_value(SecretId=secret_id)
    return json.loads(resp["SecretString"])

def lambda_handler(event, context):
    # Env vars
    secret_id = os.environ["RDS_SECRET_NAME"]              # e.g. rds!db-...
    db_host   = os.environ["DB_HOST"]                      # your RDS endpoint hostname
    db_port   = int(os.environ.get("DB_PORT", "5432"))
    db_name   = os.environ.get("DB_NAME", "retailds")
    db_schema = os.environ.get("DB_SCHEMA", "retailds")

    # Read username/password from Secrets Manager
    sec = get_secret(secret_id)
    user = sec["username"]
    pwd  = sec["password"]

    # Connect
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=user,
        password=pwd,
        connect_timeout=10
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Truncate ONLY staging tables (production style)
    cur.execute(f"""
        TRUNCATE TABLE
          {db_schema}.fact_sales_weekly_stg,
          {db_schema}.dim_store_stg,
          {db_schema}.dim_dept_stg
        RESTART IDENTITY;
    """)

    cur.close()
    conn.close()
    return {"statusCode": 200, "body": "Staging truncated"}