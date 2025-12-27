-- Connect to the right DB (ignored by DBeaver; useful in psql)
\c retailds;

-- 1) Dimension: stores (lookup for store attributes)
DROP TABLE IF EXISTS dim_store CASCADE;
CREATE TABLE dim_store (
  store_id   INT PRIMARY KEY,
  store_type CHAR(1),    -- 'A','B','C'
  store_size INT         -- square feet
);

-- 2) Dimension: departments (lookup for dept IDs)
DROP TABLE IF EXISTS dim_dept CASCADE;
CREATE TABLE dim_dept (
  dept_id INT PRIMARY KEY
);

-- 3) Fact: weekly sales + covariates at (store, dept, week)
DROP TABLE IF EXISTS fact_sales_weekly CASCADE;
CREATE TABLE fact_sales_weekly (
  store_id     INT NOT NULL REFERENCES dim_store(store_id),
  dept_id      INT NOT NULL REFERENCES dim_dept(dept_id),
  week_date    DATE NOT NULL,            -- Friday dates in Walmart data
  weekly_sales NUMERIC(12,2),            -- money-like, 2 decimals
  is_holiday   BOOLEAN,
  temperature  NUMERIC(6,2),
  fuel_price   NUMERIC(6,3),
  markdown1    NUMERIC(12,2),
  markdown2    NUMERIC(12,2),
  markdown3    NUMERIC(12,2),
  markdown4    NUMERIC(12,2),
  markdown5    NUMERIC(12,2),
  cpi          NUMERIC(10,7),
  unemployment NUMERIC(6,3),
  PRIMARY KEY (store_id, dept_id, week_date)
);

-- 4) Staging Table (Load the new data into a staging table fact_sales_weekly_stg, Run SQL to upsert into final fact table, Clear staging)
CREATE TABLE IF NOT EXISTS fact_sales_weekly_stg (
  store_id      INT,
  dept_id       INT,
  week_date     DATE,
  weekly_sales  NUMERIC(12,2),
  is_holiday    BOOLEAN,
  temperature   NUMERIC(6,2),
  fuel_price    NUMERIC(6,3),
  markdown1     NUMERIC(12,2),
  markdown2     NUMERIC(12,2),
  markdown3     NUMERIC(12,2),
  markdown4     NUMERIC(12,2),
  markdown5     NUMERIC(12,2),
  cpi           NUMERIC(12,7),
  unemployment  NUMERIC(6,3)
);

-- Helpful indexes (speed up time-based filters / joins)
CREATE INDEX IF NOT EXISTS idx_sales_week_date ON fact_sales_weekly(week_date);