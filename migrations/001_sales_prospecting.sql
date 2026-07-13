-- Migration: Sales Prospecting plugin tables
-- Two tables: runs (discovery sessions) + prospects (scored candidates per run).
-- Reuses cvc.sales_targets and cvc.sales_briefings for promote + briefing steps.

CREATE TABLE IF NOT EXISTS cvc.sales_prospect_runs (
    id              SERIAL PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'running', 'complete', 'error')),
    criteria        JSONB NOT NULL DEFAULT '{}',  -- {sectors:[], region, tech_focus}
    error_message   TEXT,
    candidate_count INT  NOT NULL DEFAULT 0,
    created_by      TEXT NOT NULL,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sales_prospect_runs_created_by ON cvc.sales_prospect_runs(created_by);
CREATE INDEX IF NOT EXISTS idx_sales_prospect_runs_status    ON cvc.sales_prospect_runs(status);

CREATE TABLE IF NOT EXISTS cvc.sales_prospects (
    id                  SERIAL PRIMARY KEY,
    run_id              INT  NOT NULL REFERENCES cvc.sales_prospect_runs(id) ON DELETE CASCADE,
    company_name        TEXT NOT NULL,
    website             TEXT,
    revenue_verified    BOOLEAN  NOT NULL DEFAULT FALSE,
    revenue_evidence    TEXT,                               -- Brave snippet/url confirming >$1bn
    score_innovation    INT,                                -- 0-100
    score_budget        INT,
    score_tech_fit      INT,
    score_overall       INT,
    tech_direction      JSONB NOT NULL DEFAULT '{}',        -- {summary, signals:[], sources:[]}
    scoring_json        JSONB NOT NULL DEFAULT '{}',        -- per-criterion rationale + sources
    status              TEXT NOT NULL DEFAULT 'scored'
                            CHECK (status IN ('scored', 'rejected', 'promoted')),
    promoted_target_id  INT,                                -- set on promote (nullable)
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, company_name)
);

CREATE INDEX IF NOT EXISTS idx_sales_prospects_run_id ON cvc.sales_prospects(run_id);
CREATE INDEX IF NOT EXISTS idx_sales_prospects_status ON cvc.sales_prospects(status);
