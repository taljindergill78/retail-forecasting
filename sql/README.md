# SQL Schema & Migrations

This folder is the **single source of truth** for the database structure used by the project.
- We **design and version** schema changes here.
- We **run** these scripts using *either* DBeaver (GUI) **or** `psql` (terminal).
- Data files (CSV/Parquet) live in **S3**, not here.

## Files
- `01_create_schema.sql` — creates core tables:
  - `dim_store` (store lookup)
  - `dim_dept` (department lookup; future-proof)
  - `fact_sales_weekly` (sales + covariates at store/dept/week)
- Future changes should be **new files**: `02_...sql`, `03_...sql`, etc.  
  Do **not** rewrite old files; keep a migration history.

## How to run (Option A: DBeaver GUI)
1. Open DBeaver → connect to your RDS Postgres (`retailds` DB).
2. Open SQL Editor → paste `01_create_schema.sql`.
3. Click **Execute** (▶).  
4. Verify tables under **Schemas → public → Tables**.  
5. (If tree doesn’t refresh, right-click **Tables → Refresh**.)

## How to run (Option B: raw SQL via psql)
```bash
# Replace <ENDPOINT> with your RDS endpoint
psql "host=<ENDPOINT> port=5432 dbname=retailds user=retailadmin sslmode=require"
\i /full/path/to/sql/01_create_schema.sql
\dt