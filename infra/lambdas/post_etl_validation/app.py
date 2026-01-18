"""
Data quality validation script for retail forecasting dataset.

This script checks:
- Row counts match expected values
- No nulls in critical columns
- Date ranges are valid
- No duplicate records
- Foreign key integrity
- Numeric columns are valid numbers
"""

import os
import sys
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging for CloudWatch Logs
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Optional: Only import boto3 if using Secrets Manager
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


def get_secret(secret_id: str, region: str = None) -> dict:
    """Get secret from AWS Secrets Manager."""
    if not BOTO3_AVAILABLE:
        raise ImportError("boto3 is required for Secrets Manager. Install with: pip install boto3")
    
    region = region or os.environ.get("AWS_REGION", "us-west-2")
    sm = boto3.client("secretsmanager", region_name=region)
    resp = sm.get_secret_value(SecretId=secret_id)
    return json.loads(resp["SecretString"])


def get_db_connection():
    """
    Get database connection from AWS Secrets Manager (Lambda always uses Secrets Manager).
    
    Environment variables (set in Lambda configuration):
    - DB_HOST: RDS endpoint
    - DB_PORT: Port (default: 5432)
    - DB_NAME: Database name (default: retailds)
    - RDS_SECRET_NAME: Secrets Manager ARN (required)
    - AWS_REGION: AWS region (default: us-west-2)
    """
    # Get connection parameters
    db_host = os.environ.get("DB_HOST")
    db_port = int(os.environ.get("DB_PORT", "5432"))
    db_name = os.environ.get("DB_NAME", "retailds")
    secret_name = os.environ.get("RDS_SECRET_NAME")
    
    if not db_host:
        raise ValueError("DB_HOST environment variable is required in Lambda configuration")
    
    if not secret_name:
        raise ValueError("RDS_SECRET_NAME environment variable is required in Lambda configuration")
    
    # Lambda always uses Secrets Manager
    if not BOTO3_AVAILABLE:
        raise ImportError("boto3 is required for Secrets Manager. Install with: pip install boto3")
    
    secret = get_secret(secret_name)
    db_user = secret.get("username") or secret.get("user")
    db_password = secret.get("password")
    
    if not db_user or not db_password:
        raise ValueError(f"Secret {secret_name} must contain 'username' and 'password' keys")
    
    # Connect to database
    return psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        connect_timeout=10
    )


def check_row_counts(conn):
    """Verify row counts match expected values."""
    print("🔍 Checking row counts...")
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    checks = [
        ("dim_store", 45),
        ("dim_dept", 81),
        ("fact_sales_weekly", 421570),
        ("vw_sales_analytics", 421570),  # View should match fact table
    ]
    
    all_passed = True
    for table_name, expected_count in checks:
        cur.execute(f"SELECT COUNT(*) as cnt FROM retailds.{table_name}")
        actual_count = cur.fetchone()["cnt"]
        
        if actual_count == expected_count:
            print(f"  ✅ {table_name}: {actual_count} rows (expected {expected_count})")
        else:
            print(f"  ❌ {table_name}: {actual_count} rows (expected {expected_count})")
            all_passed = False
    
    cur.close()
    return all_passed


def check_null_values(conn):
    """Check for nulls in critical columns."""
    print("\n🔍 Checking for null values in critical columns...")
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    checks = [
        ("fact_sales_weekly", "store_id"),
        ("fact_sales_weekly", "dept_id"),
        ("fact_sales_weekly", "week_date"),
        ("fact_sales_weekly", "weekly_sales"),
    ]
    
    all_passed = True
    for table_name, column_name in checks:
        cur.execute(f"""
            SELECT COUNT(*) as cnt 
            FROM retailds.{table_name} 
            WHERE {column_name} IS NULL
        """)
        null_count = cur.fetchone()["cnt"]
        
        if null_count == 0:
            print(f"  ✅ {table_name}.{column_name}: No nulls")
        else:
            print(f"  ❌ {table_name}.{column_name}: {null_count} null values found!")
            all_passed = False
    
    cur.close()
    return all_passed


def check_duplicates(conn):
    """Check for duplicate (store_id, dept_id, week_date) combinations."""
    print("\n🔍 Checking for duplicate records...")
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("""
        SELECT store_id, dept_id, week_date, COUNT(*) as cnt
        FROM retailds.fact_sales_weekly
        GROUP BY store_id, dept_id, week_date
        HAVING COUNT(*) > 1
    """)
    
    duplicates = cur.fetchall()
    
    if len(duplicates) == 0:
        print("  ✅ No duplicate (store_id, dept_id, week_date) combinations")
        cur.close()
        return True
    else:
        print(f"  ❌ Found {len(duplicates)} duplicate combinations!")
        for dup in duplicates[:5]:  # Show first 5
            print(f"     Store {dup['store_id']}, Dept {dup['dept_id']}, Date {dup['week_date']}: {dup['cnt']} rows")
        cur.close()
        return False


def check_foreign_keys(conn):
    """Verify foreign key integrity."""
    print("\n🔍 Checking foreign key integrity...")
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Check if all store_ids in fact table exist in dim_store
    cur.execute("""
        SELECT COUNT(*) as cnt
        FROM retailds.fact_sales_weekly f
        LEFT JOIN retailds.dim_store s ON f.store_id = s.store_id
        WHERE s.store_id IS NULL
    """)
    orphan_stores = cur.fetchone()["cnt"]
    
    # Check if all dept_ids in fact table exist in dim_dept
    cur.execute("""
        SELECT COUNT(*) as cnt
        FROM retailds.fact_sales_weekly f
        LEFT JOIN retailds.dim_dept d ON f.dept_id = d.dept_id
        WHERE d.dept_id IS NULL
    """)
    orphan_depts = cur.fetchone()["cnt"]
    
    all_passed = True
    if orphan_stores == 0:
        print("  ✅ All store_ids in fact table exist in dim_store")
    else:
        print(f"  ❌ Found {orphan_stores} orphaned store_ids!")
        all_passed = False
    
    if orphan_depts == 0:
        print("  ✅ All dept_ids in fact table exist in dim_dept")
    else:
        print(f"  ❌ Found {orphan_depts} orphaned dept_ids!")
        all_passed = False
    
    cur.close()
    return all_passed


def check_date_range(conn):
    """Verify date range is reasonable."""
    print("\n🔍 Checking date range...")
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("""
        SELECT 
            MIN(week_date) as min_date,
            MAX(week_date) as max_date
        FROM retailds.fact_sales_weekly
    """)
    result = cur.fetchone()
    min_date = result["min_date"]
    max_date = result["max_date"]
    
    print(f"  📅 Date range: {min_date} to {max_date}")
    
    # Basic sanity: dates should be between 2010 and 2015 (Walmart dataset range)
    if min_date.year >= 2010 and max_date.year <= 2015:
        print("  ✅ Date range is within expected bounds (2010-2015)")
        cur.close()
        return True
    else:
        print(f"  ⚠️  Date range might be unexpected (expected 2010-2015)")
        cur.close()
        return True  # Not a hard failure, just a warning


def check_numeric_ranges(conn):
    """Check that numeric columns have reasonable values."""
    print("\n🔍 Checking numeric value ranges...")
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Check for negative sales (shouldn't happen, but could indicate data issues)
    cur.execute("""
        SELECT COUNT(*) as cnt
        FROM retailds.fact_sales_weekly
        WHERE weekly_sales < 0
    """)
    negative_sales = cur.fetchone()["cnt"]
    
    if negative_sales == 0:
        print("  ✅ No negative weekly_sales values")
    else:
        print(f"  ⚠️  Found {negative_sales} rows with negative weekly_sales")
    
    cur.close()
    return True  # Not a hard failure


def run_validation():
    """Run all validation checks and return results."""
    logger.info("=" * 60)
    logger.info("📊 Retail Forecasting Data Quality Validation")
    logger.info("=" * 60)
    
    try:
        conn = get_db_connection()
        logger.info("✅ Connected to database")
        
        results = []
        results.append(("Row Counts", check_row_counts(conn)))
        results.append(("Null Values", check_null_values(conn)))
        results.append(("Duplicates", check_duplicates(conn)))
        results.append(("Foreign Keys", check_foreign_keys(conn)))
        results.append(("Date Range", check_date_range(conn)))
        results.append(("Numeric Ranges", check_numeric_ranges(conn)))
        
        conn.close()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📋 Validation Summary")
        logger.info("=" * 60)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        summary_list = []
        for check_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            message = f"{status} - {check_name}"
            logger.info(message)
            summary_list.append({"check": check_name, "passed": result})
        
        logger.info(f"\nResult: {passed}/{total} checks passed")
        
        all_passed = passed == total
        if all_passed:
            logger.info("🎉 All validations passed! Data is ready for ML.")
        else:
            logger.warning("⚠️  Some validations failed. Please review the issues above.")
        
        return {
            "all_passed": all_passed,
            "passed": passed,
            "total": total,
            "checks": summary_list
        }
            
    except Exception as e:
        error_msg = f"❌ Error during validation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


def lambda_handler(event, context):
    """
    Lambda handler for data quality validation.
    
    This function is called by AWS Lambda after ETL completes.
    It validates data quality and returns results as JSON.
    
    Args:
        event: Lambda event (usually empty dict for scheduled triggers)
        context: Lambda context
        
    Returns:
        dict: {
            "statusCode": 200 or 500,
            "body": JSON string with validation results
        }
    """
    try:
        results = run_validation()
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Validation completed successfully",
                "validation_passed": results["all_passed"],
                "checks_passed": results["passed"],
                "total_checks": results["total"],
                "details": results["checks"]
            })
        }
        
    except Exception as e:
        logger.error(f"Lambda execution failed: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "message": "Validation failed with error",
                "error": str(e)
            })
        }