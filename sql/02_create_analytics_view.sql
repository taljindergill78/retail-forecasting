-- ============================================================
-- Analytics-ready view for ML: fact + store + dept in one query
-- Run this inside database "retailds", schema "retailds"
-- ============================================================

CREATE OR REPLACE VIEW retailds.vw_sales_analytics AS
SELECT
    f.store_id,
    s.store_type,
    s.store_size,
    f.dept_id,
    f.week_date,
    f.weekly_sales,
    f.isholiday,
    f.temperature,
    f.fuel_price,
    f.markdown1,
    f.markdown2,
    f.markdown3,
    f.markdown4,
    f.markdown5,
    f.cpi,
    f.unemployment
FROM retailds.fact_sales_weekly f
JOIN retailds.dim_store s ON f.store_id = s.store_id
JOIN retailds.dim_dept d ON f.dept_id = d.dept_id;