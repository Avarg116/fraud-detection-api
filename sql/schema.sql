-- Every prediction the API makes, one row each.
-- Runs automatically the first time the Postgres container starts (empty data volume).

CREATE TABLE IF NOT EXISTS predictions (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at     TIMESTAMPTZ      NOT NULL DEFAULT now(),

    amount         NUMERIC(12, 2)   NOT NULL CHECK (amount >= 0),
    features       JSONB            NOT NULL,

    fraud_score    DOUBLE PRECISION NOT NULL CHECK (fraud_score BETWEEN 0 AND 1),
    decision       TEXT             NOT NULL CHECK (decision IN ('APPROVE', 'FLAG')),
    model_version  TEXT             NOT NULL,
    threshold      DOUBLE PRECISION NOT NULL CHECK (threshold BETWEEN 0 AND 1)
);

-- last hour/day queries.
CREATE INDEX IF NOT EXISTS predictions_created_at_idx ON predictions (created_at);

-- Review queue. Only FLAG rows (~0.2% of traffic) are indexed
CREATE INDEX IF NOT EXISTS predictions_flagged_idx ON predictions (created_at) WHERE decision = 'FLAG';
