-- ============================================================
-- A) CREATE DATABASE (run while connected to "postgres")
-- ============================================================

-- Optional: drop if you want a clean reset (dangerous, deletes everything)
-- DROP DATABASE IF EXISTS retailds;

CREATE DATABASE retailds;


-- ============================================================
-- B) SCHEMA + TABLES (run inside database "retailds")
-- ============================================================

-- 1) Create schema if not exists
CREATE SCHEMA IF NOT EXISTS retailds;

-- 2) Make sure the default schema for this session is retailds
SET search_path TO retailds;

-- (Optional but recommended) Confirm where tables will be created
-- SHOW search_path;

-- 3) Drop in correct dependency order (fact first, then dims)
DROP TABLE IF EXISTS retailds.fact_sales_weekly_stg CASCADE;
DROP TABLE IF EXISTS retailds.fact_sales_weekly CASCADE;
DROP TABLE IF EXISTS retailds.dim_dept CASCADE;
DROP TABLE IF EXISTS retailds.dim_store CASCADE;

-- 4) Dimension: stores
CREATE TABLE retailds.dim_store (
  store_id   INT PRIMARY KEY,
  store_type CHAR(1),    -- 'A','B','C'
  store_size INT         -- square feet
);

-- 5) Dimension: departments
CREATE TABLE retailds.dim_dept (
  dept_id INT PRIMARY KEY
);

-- 6) Fact: weekly sales + covariates at (store, dept, week)
CREATE TABLE retailds.fact_sales_weekly (
  store_id     INT NOT NULL REFERENCES retailds.dim_store(store_id),
  dept_id      INT NOT NULL REFERENCES retailds.dim_dept(dept_id),
  week_date    DATE NOT NULL,            -- Friday dates in Walmart data
  weekly_sales NUMERIC(12,2),            -- money-like, 2 decimals
  isholiday    BOOLEAN,
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

-- 7) Staging table
CREATE TABLE retailds.fact_sales_weekly_stg (
  store_id      INT,
  dept_id       INT,
  week_date     DATE,
  weekly_sales  NUMERIC(12,2),
  isholiday     BOOLEAN,
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

-- 8) Helpful index
CREATE INDEX IF NOT EXISTS idx_sales_week_date
ON retailds.fact_sales_weekly(week_date);