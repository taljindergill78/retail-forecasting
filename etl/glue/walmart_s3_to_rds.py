import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F

args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# -----------------------------
# 1) Inputs in S3
# -----------------------------
S3_STORES = "s3://retail-ml-taljinder-2025-new/raw/walmart/stores/stores.csv"
S3_FEATURES = "s3://retail-ml-taljinder-2025-new/raw/walmart/features/features.csv"
S3_TRAIN = "s3://retail-ml-taljinder-2025-new/raw/walmart/train/train.csv"

# -----------------------------
# 2) Targets in Postgres
# -----------------------------
DB_NAME = "retailds"
SCHEMA = "retailds"

# STAGING TABLES ONLY
DIM_STORE_STG = "dim_store_stg"
DIM_DEPT_STG = "dim_dept_stg"
FACT_STG = "fact_sales_weekly_stg"

# IMPORTANT: replace with your actual Glue connection name
CONN_NAME = "glue-retail-postgres-connection"

def norm_cols(df):
    for c in df.columns:
        df = df.withColumnRenamed(c, c.strip().lower().replace(" ", "_"))
    return df

def parse_date(colname: str):
    return F.to_date(F.col(colname), "yyyy-MM-dd")

def to_bool(colname: str):
    c = F.col(colname)
    return (
        F.when(c.isNull(), F.lit(None).cast("boolean"))
         .when((c == True) | (c == False), c.cast("boolean"))
         .when(c.cast("string").isin("true", "false", "True", "False"), c.cast("boolean"))
         .when(c.cast("string").isin("1", "0"), (c.cast("int") == 1).cast("boolean"))
         .otherwise(c.cast("boolean"))
    )

def write_table(df, table, mode="append"):
    dyf = DynamicFrame.fromDF(df, glueContext, f"dyf_{table}")
    glueContext.write_dynamic_frame.from_jdbc_conf(
        frame=dyf,
        catalog_connection=CONN_NAME,
        connection_options={
            "database": DB_NAME,
            "dbtable": f"{SCHEMA}.{table}",
        },
        transformation_ctx=f"write_{table}",
    )

# -----------------------------
# 3) Read CSVs
# -----------------------------
stores = norm_cols(spark.read.option("header", "true").csv(S3_STORES))
features = norm_cols(spark.read.option("header", "true").csv(S3_FEATURES))
train = norm_cols(spark.read.option("header", "true").csv(S3_TRAIN))

# -----------------------------
# 4) Transform
# -----------------------------
stores_df = (
    stores.select(
        F.col("store").cast("int").alias("store_id"),
        F.col("type").cast("string").substr(1, 1).alias("store_type"),
        F.col("size").cast("int").alias("store_size"),
    )
    .dropDuplicates(["store_id"])
)

features_df = (
    features.select(
        F.col("store").cast("int").alias("store_id"),
        parse_date("date").alias("week_date"),
        to_bool("isholiday").alias("isholiday"),
        F.col("temperature").cast("double").alias("temperature"),
        F.col("fuel_price").cast("double").alias("fuel_price"),
        F.col("markdown1").cast("double").alias("markdown1"),
        F.col("markdown2").cast("double").alias("markdown2"),
        F.col("markdown3").cast("double").alias("markdown3"),
        F.col("markdown4").cast("double").alias("markdown4"),
        F.col("markdown5").cast("double").alias("markdown5"),
        F.col("cpi").cast("double").alias("cpi"),
        F.col("unemployment").cast("double").alias("unemployment"),
    )
)

train_df = (
    train.select(
        F.col("store").cast("int").alias("store_id"),
        F.col("dept").cast("int").alias("dept_id"),
        parse_date("date").alias("week_date"),
        F.col("weekly_sales").cast("double").alias("weekly_sales"),
        to_bool("isholiday").alias("isholiday"),
    )
)

depts_df = train_df.select("dept_id").dropDuplicates(["dept_id"])

fact_rows = (
    train_df.join(features_df, on=["store_id", "week_date", "isholiday"], how="left")
    .select(
        "store_id",
        "dept_id",
        "week_date",
        "weekly_sales",
        "isholiday",
        "temperature",
        "fuel_price",
        "markdown1",
        "markdown2",
        "markdown3",
        "markdown4",
        "markdown5",
        "cpi",
        "unemployment",
    )
    .filter(
        F.col("store_id").isNotNull()
        & F.col("dept_id").isNotNull()
        & F.col("week_date").isNotNull()
    )
)

# -----------------------------
# 5) Load STAGING ONLY
# -----------------------------
write_table(stores_df, DIM_STORE_STG, mode="append")
write_table(depts_df, DIM_DEPT_STG, mode="append")
write_table(fact_rows, FACT_STG, mode="append")

job.commit()