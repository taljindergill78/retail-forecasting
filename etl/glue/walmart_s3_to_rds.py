import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F

# -----------------------------
# 0) Glue boilerplate
# -----------------------------
args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# -----------------------------
# 1) Inputs in S3
# -----------------------------
S3_STORES   = "s3://retail-ml-taljinder-2025/raw/walmart/stores/stores.csv"
S3_FEATURES = "s3://retail-ml-taljinder-2025/raw/walmart/features/features.csv"
S3_TRAIN    = "s3://retail-ml-taljinder-2025/raw/walmart/train/train.csv"

# -----------------------------
# 2) Postgres targets
# -----------------------------
DB_NAME  = "retailds"   # database name
SCHEMA   = "retailds"   # schema name (your tables live here)
CONN_NAME = "glue-retail-postgres-connection"

DIM_STORE_TABLE = "dim_store"
DIM_DEPT_TABLE  = "dim_dept"
FACT_TABLE      = "fact_sales_weekly"
FACT_STG_TABLE  = "fact_sales_weekly_stg"

# -----------------------------
# 3) Helpers
# -----------------------------
def norm(df):
    for c in df.columns:
        df = df.withColumnRenamed(c, c.strip().lower().replace(" ", "_"))
    return df

def parse_walmart_date(colname: str):
    c = F.col(colname)
    return F.coalesce(
        F.to_date(c, "dd/MM/yy"),
        F.to_date(c, "yyyy-MM-dd"),
        F.to_date(c)
    )

def to_decimal_safe(col, decimal_type: str):
    cleaned = F.upper(F.trim(F.col(col)))
    return (
        F.when((cleaned == "NA") | (cleaned == "N/A") | (cleaned == ""), None)
         .otherwise(F.regexp_replace(F.trim(F.col(col)), ",", ""))
         .cast(decimal_type)
    )

def write_table(df, table, mode="append", preactions="", postactions=""):
    effective_preactions = preactions
    if mode == "overwrite" and not preactions:
        effective_preactions = f"TRUNCATE TABLE {SCHEMA}.{table};"

    dyf = DynamicFrame.fromDF(df, glueContext, f"dyf_{table}")

    conn_opts = {
        "database": DB_NAME,
        "dbtable": f"{SCHEMA}.{table}",
    }
    if effective_preactions:
        conn_opts["preactions"] = effective_preactions
    if postactions:
        conn_opts["postactions"] = postactions

    glueContext.write_dynamic_frame.from_jdbc_conf(
        frame=dyf,
        catalog_connection=CONN_NAME,
        connection_options=conn_opts,
        transformation_ctx=f"write_{table}"
    )

def exec_sql(sql: str, ctx: str, schema_for_dummy):
    """
    Run SQL via postactions using a 0-row frame with a known schema.
    """
    dyf0 = DynamicFrame.fromDF(schema_for_dummy.limit(0), glueContext, f"dyf0_{ctx}")
    glueContext.write_dynamic_frame.from_jdbc_conf(
        frame=dyf0,
        catalog_connection=CONN_NAME,
        connection_options={
            "database": DB_NAME,
            "dbtable": f"{SCHEMA}.{FACT_STG_TABLE}",
            "postactions": sql
        },
        transformation_ctx=ctx
    )

# -----------------------------
# 4) Read CSVs
# -----------------------------
stores_df   = spark.read.option("header", "true").csv(S3_STORES)
features_df = spark.read.option("header", "true").csv(S3_FEATURES)
train_df    = spark.read.option("header", "true").csv(S3_TRAIN)

stores   = norm(stores_df)
features = norm(features_df)
train    = norm(train_df)

# -----------------------------
# 5) Build dims + fact
# -----------------------------
dim_store = (
    stores
      .withColumn("store_id", F.col("store").cast("int"))
      .withColumn("store_type", F.col("type"))
      .withColumn("store_size", F.col("size").cast("int"))
      .select("store_id", "store_type", "store_size")
      .dropDuplicates(["store_id"])
)

train_clean = (
    train
      .withColumn("store_id", F.col("store").cast("int"))
      .withColumn("dept_id", F.col("dept").cast("int"))
      .withColumn("week_date", parse_walmart_date("date"))
      .withColumn("weekly_sales", to_decimal_safe("weekly_sales", "decimal(12,2)"))
      .withColumn("isholiday", F.col("isholiday").cast("boolean"))
      .select("store_id", "dept_id", "week_date", "weekly_sales", "isholiday")
)

features_clean = (
    features
      .withColumn("store_id", F.col("store").cast("int"))
      .withColumn("week_date", parse_walmart_date("date"))
      .withColumn("temperature", to_decimal_safe("temperature", "decimal(6,2)"))
      .withColumn("fuel_price",  to_decimal_safe("fuel_price",  "decimal(6,3)"))
      .withColumn("markdown1",   to_decimal_safe("markdown1",   "decimal(12,2)"))
      .withColumn("markdown2",   to_decimal_safe("markdown2",   "decimal(12,2)"))
      .withColumn("markdown3",   to_decimal_safe("markdown3",   "decimal(12,2)"))
      .withColumn("markdown4",   to_decimal_safe("markdown4",   "decimal(12,2)"))
      .withColumn("markdown5",   to_decimal_safe("markdown5",   "decimal(12,2)"))
      .withColumn("cpi",         to_decimal_safe("cpi",         "decimal(12,7)"))
      .withColumn("unemployment",to_decimal_safe("unemployment","decimal(6,3)"))
      .withColumn("isholiday",   F.col("isholiday").cast("boolean"))
      .select(
          "store_id", "week_date", "isholiday",
          "temperature", "fuel_price",
          "markdown1", "markdown2", "markdown3", "markdown4", "markdown5",
          "cpi", "unemployment"
      )
)

# Fail fast if date parsing broke
train_null_dates = train_clean.filter(F.col("week_date").isNull()).count()
features_null_dates = features_clean.filter(F.col("week_date").isNull()).count()
if train_null_dates > 0 or features_null_dates > 0:
    raise Exception(
        f"Date parsing produced nulls. train_null_dates={train_null_dates}, "
        f"features_null_dates={features_null_dates}."
    )

dim_dept = train_clean.select("dept_id").dropDuplicates(["dept_id"])

fact_rows = (
    train_clean.alias("t")
      .join(features_clean.alias("f"), on=["store_id", "week_date"], how="left")
      .select(
          F.col("t.store_id"),
          F.col("t.dept_id"),
          F.col("t.week_date"),
          F.col("t.weekly_sales"),
          F.coalesce(F.col("t.isholiday"), F.col("f.isholiday")).alias("isholiday"),
          F.col("f.temperature"),
          F.col("f.fuel_price"),
          F.col("f.markdown1"),
          F.col("f.markdown2"),
          F.col("f.markdown3"),
          F.col("f.markdown4"),
          F.col("f.markdown5"),
          F.col("f.cpi"),
          F.col("f.unemployment")
      )
      .filter(
          F.col("store_id").isNotNull() &
          F.col("dept_id").isNotNull() &
          F.col("week_date").isNotNull()
      )
)

# -----------------------------
# 6) Hard reset tables first (prevents PK collisions)
# -----------------------------
reset_sql = f"""
TRUNCATE TABLE
  {SCHEMA}.{FACT_STG_TABLE},
  {SCHEMA}.{FACT_TABLE},
  {SCHEMA}.{DIM_DEPT_TABLE},
  {SCHEMA}.{DIM_STORE_TABLE}
;
"""
exec_sql(reset_sql, "exec_reset", fact_rows)

# -----------------------------
# 7) Load dims
# -----------------------------
write_table(dim_store, DIM_STORE_TABLE, mode="append")
write_table(dim_dept,  DIM_DEPT_TABLE,  mode="append")

# -----------------------------
# 8) Load staging + upsert + clear staging
# -----------------------------
write_table(fact_rows, FACT_STG_TABLE, mode="overwrite")

upsert_sql = f"""
INSERT INTO {SCHEMA}.{FACT_TABLE} (
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
FROM {SCHEMA}.{FACT_STG_TABLE}
ON CONFLICT (store_id, dept_id, week_date)
DO UPDATE SET
  weekly_sales = EXCLUDED.weekly_sales,
  isholiday   = EXCLUDED.isholiday,
  temperature  = EXCLUDED.temperature,
  fuel_price   = EXCLUDED.fuel_price,
  markdown1    = EXCLUDED.markdown1,
  markdown2    = EXCLUDED.markdown2,
  markdown3    = EXCLUDED.markdown3,
  markdown4    = EXCLUDED.markdown4,
  markdown5    = EXCLUDED.markdown5,
  cpi          = EXCLUDED.cpi,
  unemployment = EXCLUDED.unemployment;
"""
exec_sql(upsert_sql, "exec_upsert", fact_rows)
exec_sql(f"TRUNCATE TABLE {SCHEMA}.{FACT_STG_TABLE};", "exec_truncate_stg", fact_rows)

job.commit()