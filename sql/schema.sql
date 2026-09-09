-- =============================================================================
-- PREDICT and DECIDE Data Layer Schema for PostgreSQL
-- Grain compatibility: Integrates with existing Power BI (Customer Sales Insights)
-- =============================================================================

-- Ensure UUID or necessary extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -----------------------------------------------------------------------------
-- 1. Sales_Forecast
-- Grain: customer_id × forecast_month
-- Stores predicted sales amount alongside upper and lower confidence intervals.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sales_forecast (
    forecast_id             BIGSERIAL PRIMARY KEY,
    customer_id             VARCHAR(64) NOT NULL,
    forecast_month          DATE NOT NULL, -- First day of the forecast month (YYYY-MM-01)
    predicted_sales         NUMERIC(14, 2) NOT NULL,
    lower_bound_ci          NUMERIC(14, 2) NOT NULL, -- e.g., 80% or 95% confidence lower interval
    upper_bound_ci          NUMERIC(14, 2) NOT NULL, -- e.g., 80% or 95% confidence upper interval
    model_version           VARCHAR(50) DEFAULT 'prophet-v1.0',
    created_at              TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_sales_forecast_cust_month UNIQUE (customer_id, forecast_month)
);

CREATE INDEX IF NOT EXISTS idx_sales_forecast_cust ON sales_forecast (customer_id);
CREATE INDEX IF NOT EXISTS idx_sales_forecast_month ON sales_forecast (forecast_month);


-- -----------------------------------------------------------------------------
-- 2. Customer_Risk
-- Grain: customer_id (latest assessment snapshot / daily partition)
-- Stores churn/attrition risk score (0-100), revenue at risk, and top SHAP reasons.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS customer_risk (
    risk_id                 BIGSERIAL PRIMARY KEY,
    customer_id             VARCHAR(64) NOT NULL,
    as_of_date              DATE NOT NULL,
    risk_score              NUMERIC(5, 2) NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
    risk_level              VARCHAR(20) NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    revenue_at_risk         NUMERIC(14, 2) NOT NULL DEFAULT 0.00,
    top_reasons             JSONB NOT NULL DEFAULT '[]'::jsonb, -- Top SHAP feature contributions e.g. [{"feature": "days_since_last_order", "impact": 0.32}]
    model_version           VARCHAR(50) DEFAULT 'xgboost-shap-v1.0',
    created_at              TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_customer_risk_cust_date UNIQUE (customer_id, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_customer_risk_cust ON customer_risk (customer_id);
CREATE INDEX IF NOT EXISTS idx_customer_risk_date ON customer_risk (as_of_date);
CREATE INDEX IF NOT EXISTS idx_customer_risk_level ON customer_risk (risk_level);


-- -----------------------------------------------------------------------------
-- 3. Sales_Opportunity
-- Grain: customer_id × opportunity_type
-- Captures upsell/cross-sell opportunities, potential revenue, and win probability.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sales_opportunity (
    opportunity_id          BIGSERIAL PRIMARY KEY,
    customer_id             VARCHAR(64) NOT NULL,
    as_of_date              DATE NOT NULL,
    opportunity_type        VARCHAR(64) NOT NULL, -- e.g. 'CROSS_SELL_CATEGORY_A', 'VOLUME_UPGRADE', 'REENGAGEMENT'
    potential_revenue       NUMERIC(14, 2) NOT NULL,
    win_probability         NUMERIC(4, 3) NOT NULL CHECK (win_probability >= 0.000 AND win_probability <= 1.000),
    expected_value          NUMERIC(14, 2) GENERATED ALWAYS AS (potential_revenue * win_probability) STORED,
    details                 JSONB DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_sales_opp_cust_type_date UNIQUE (customer_id, opportunity_type, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_sales_opp_cust ON sales_opportunity (customer_id);
CREATE INDEX IF NOT EXISTS idx_sales_opp_date ON sales_opportunity (as_of_date);


-- -----------------------------------------------------------------------------
-- 4. Anomaly_Output
-- Grain: entity_id (customer_id or product_id) × entity_type × date × metric
-- Captures statistical sales spikes or sudden drop anomalies.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS anomaly_output (
    anomaly_id              BIGSERIAL PRIMARY KEY,
    entity_type             VARCHAR(32) NOT NULL, -- 'CUSTOMER', 'PRODUCT', 'REGION'
    entity_id               VARCHAR(64) NOT NULL,
    anomaly_date            DATE NOT NULL,
    metric_name             VARCHAR(64) NOT NULL, -- 'DAILY_SALES', 'ORDER_FREQUENCY', 'AVG_ORDER_VALUE'
    actual_value            NUMERIC(14, 2) NOT NULL,
    expected_value          NUMERIC(14, 2) NOT NULL,
    deviation               NUMERIC(14, 2) NOT NULL, -- (actual - expected)
    z_score                 NUMERIC(6, 2) NOT NULL,
    severity                VARCHAR(20) NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    anomaly_type            VARCHAR(32) NOT NULL CHECK (anomaly_type IN ('SPIKE', 'DROP')),
    created_at              TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_anomaly_entity_metric_date UNIQUE (entity_type, entity_id, metric_name, anomaly_date)
);

CREATE INDEX IF NOT EXISTS idx_anomaly_entity ON anomaly_output (entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_date ON anomaly_output (anomaly_date);
CREATE INDEX IF NOT EXISTS idx_anomaly_severity ON anomaly_output (severity);


-- -----------------------------------------------------------------------------
-- 5. Decision_Recommendation
-- Grain: customer_id × issue_code × date (DECIDE layer actionable output)
-- Rules-based prescriptive guidance for sales reps and account managers.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS decision_recommendation (
    recommendation_id       BIGSERIAL PRIMARY KEY,
    customer_id             VARCHAR(64) NOT NULL,
    as_of_date              DATE NOT NULL,
    issue_code              VARCHAR(64) NOT NULL, -- e.g. 'CHURN_RISK_HIGH', 'ORDER_DROP_ANOMALY', 'CROSS_SELL_READY'
    issue_description       TEXT NOT NULL,
    recommended_action      TEXT NOT NULL,
    expected_impact         VARCHAR(128) NOT NULL, -- e.g. 'Preserve $15,000 monthly ARR'
    priority                VARCHAR(20) NOT NULL CHECK (priority IN ('LOW', 'MEDIUM', 'HIGH', 'URGENT')),
    status                  VARCHAR(30) NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'IN_REVIEW', 'ACCEPTED', 'DISMISSED', 'COMPLETED')),
    action_metadata         JSONB DEFAULT '{}'::jsonb, -- Supporting context/metrics
    created_at              TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_decision_rec_cust_issue_date UNIQUE (customer_id, issue_code, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_decision_rec_cust ON decision_recommendation (customer_id);
CREATE INDEX IF NOT EXISTS idx_decision_rec_priority ON decision_recommendation (priority);
CREATE INDEX IF NOT EXISTS idx_decision_rec_status ON decision_recommendation (status);


-- -----------------------------------------------------------------------------
-- 6. NextPurchase_Prediction
-- Grain: customer_id × as_of_date
-- Interval-based predictive next transaction date and expected basket value.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS next_purchase_prediction (
    prediction_id           BIGSERIAL PRIMARY KEY,
    customer_id             VARCHAR(64) NOT NULL,
    as_of_date              DATE NOT NULL,
    expected_purchase_date  DATE NOT NULL,
    expected_purchase_value NUMERIC(14, 2) NOT NULL,
    confidence_score        NUMERIC(4, 3) NOT NULL CHECK (confidence_score >= 0.000 AND confidence_score <= 1.000),
    days_until_next_purchase INT NOT NULL,
    model_version           VARCHAR(50) DEFAULT 'interval-rfm-v1.0',
    created_at              TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_next_purchase_cust_date UNIQUE (customer_id, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_next_purchase_cust ON next_purchase_prediction (customer_id);
CREATE INDEX IF NOT EXISTS idx_next_purchase_exp_date ON next_purchase_prediction (expected_purchase_date);
