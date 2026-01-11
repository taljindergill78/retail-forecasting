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

    try:
        # Merge dim_store
        cur.execute("""
            INSERT INTO retailds.dim_store (store_id, store_type, store_size)
            SELECT DISTINCT store_id, store_type, store_size
            FROM retailds.dim_store_stg
            ON CONFLICT (store_id)
            DO UPDATE SET
              store_type = EXCLUDED.store_type,
              store_size = EXCLUDED.store_size;
        """)

        # Merge dim_dept
        cur.execute("""
            INSERT INTO retailds.dim_dept (dept_id)
            SELECT DISTINCT dept_id
            FROM retailds.dim_dept_stg
            ON CONFLICT (dept_id)
            DO NOTHING;
        """)

        # Upsert fact
        cur.execute("""
            INSERT INTO retailds.fact_sales_weekly (
              store_id, dept_id, week_date, weekly_sales, isholiday,
              temperature, fuel_price,
              markdown1, markdown2, markdown3, markdown4, markdown5,
              cpi, unemployment
            )
            SELECT
              store_id, dept_id, week_date, weekly_sales, isholiday,
              temperature, fuel_price,
              markdown1, markdown2, markdown3, markdown4, markdown5,
              cpi, unemployment
            FROM retailds.fact_sales_weekly_stg
            ON CONFLICT (store_id, dept_id, week_date)
            DO UPDATE SET
              weekly_sales = EXCLUDED.weekly_sales,
              isholiday = EXCLUDED.isholiday,
              temperature = EXCLUDED.temperature,
              fuel_price = EXCLUDED.fuel_price,
              markdown1 = EXCLUDED.markdown1,
              markdown2 = EXCLUDED.markdown2,
              markdown3 = EXCLUDED.markdown3,
              markdown4 = EXCLUDED.markdown4,
              markdown5 = EXCLUDED.markdown5,
              cpi = EXCLUDED.cpi,
              unemployment = EXCLUDED.unemployment;
        """)

        # Clear staging
        cur.execute("""
            TRUNCATE TABLE
              retailds.fact_sales_weekly_stg,
              retailds.dim_store_stg,
              retailds.dim_dept_stg
            RESTART IDENTITY;
        """)

        conn.commit()
        return {"statusCode": 200, "body": "Upsert complete, staging cleared"}

    except Exception:
        conn.rollback()
        raise

    finally:
        cur.close()
        conn.close()