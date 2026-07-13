-- Migration: Add location fields to sales_prospects
-- Lets local teams see where each prospect is headquartered (granular locality).
-- country + hq_location are populated best-effort by the discovery LLM.

ALTER TABLE cvc.sales_prospects ADD COLUMN IF NOT EXISTS country     TEXT;
ALTER TABLE cvc.sales_prospects ADD COLUMN IF NOT EXISTS hq_location TEXT;

CREATE INDEX IF NOT EXISTS idx_sales_prospects_country ON cvc.sales_prospects(country);
