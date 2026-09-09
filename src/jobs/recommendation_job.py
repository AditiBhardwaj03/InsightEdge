"""
DECIDE Layer: Rules-based Recommendation Engine.
Reads external YAML config and evaluates conditions over PREDICT outputs
to generate actionable customer-level decision recommendations.
"""
import os
import yaml
import pandas as pd
from datetime import date
from string import Template
from src.common.config import settings
from src.common.db import execute_query, upsert_dataframe
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_rules(config_path: str = None) -> list:
    path = config_path or settings.rules_config_path
    if not os.path.exists(path):
        # Check relative fallback
        alt_path = os.path.join(os.path.dirname(__file__), "../../config/recommendation_rules.yaml")
        if os.path.exists(alt_path):
            path = alt_path
        else:
            logger.warning(f"Config file {path} not found. Using default empty rules.")
            return []
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        return data.get("rules", [])

def fetch_predict_layer_consolidated() -> pd.DataFrame:
    """
    Joins the latest daily outputs from the PREDICT tables:
    customer_risk, anomaly_output, sales_opportunity, and next_purchase_prediction.
    """
    query = """
    WITH latest_risk AS (
        SELECT customer_id, risk_score, risk_level, revenue_at_risk, as_of_date
        FROM customer_risk
        WHERE as_of_date = CURRENT_DATE
    ),
    latest_anomaly AS (
        SELECT entity_id as customer_id, severity, anomaly_type, deviation, z_score
        FROM anomaly_output
        WHERE anomaly_date >= CURRENT_DATE - INTERVAL '3 days' AND entity_type = 'CUSTOMER'
    ),
    latest_opp AS (
        SELECT customer_id, opportunity_type, potential_revenue, win_probability, expected_value
        FROM sales_opportunity
        WHERE as_of_date = CURRENT_DATE
    ),
    latest_np AS (
        SELECT customer_id, expected_purchase_date, expected_purchase_value, confidence_score, days_until_next_purchase
        FROM next_purchase_prediction
        WHERE as_of_date = CURRENT_DATE
    )
    SELECT 
        COALESCE(r.customer_id, a.customer_id, o.customer_id, n.customer_id) as customer_id,
        r.risk_score,
        r.risk_level,
        r.revenue_at_risk,
        a.severity as anomaly_severity,
        a.anomaly_type,
        a.deviation,
        o.opportunity_type,
        o.potential_revenue,
        o.win_probability,
        o.expected_value,
        n.expected_purchase_date,
        n.expected_purchase_value,
        n.days_until_next_purchase,
        n.confidence_score
    FROM latest_risk r
    FULL OUTER JOIN latest_anomaly a ON r.customer_id = a.customer_id
    FULL OUTER JOIN latest_opp o ON COALESCE(r.customer_id, a.customer_id) = o.customer_id
    FULL OUTER JOIN latest_np n ON COALESCE(r.customer_id, a.customer_id, o.customer_id) = n.customer_id;
    """
    try:
        df = execute_query(query)
        if not df.empty:
            return df
    except Exception as e:
        logger.warning(f"Could not read from database ({e}), building consolidated dataset from simulated job outputs.")

    # Fallback to in-memory joined data
    from src.jobs.risk_job import run_risk_scoring
    from src.jobs.anomaly_job import run_anomaly_detection
    from src.jobs.next_purchase_job import run_next_purchase_prediction

    r_df = run_risk_scoring()
    a_df = run_anomaly_detection()
    n_df = run_next_purchase_prediction()

    r_df = r_df.rename(columns={"risk_score": "risk_score", "revenue_at_risk": "revenue_at_risk"})
    a_df = a_df[a_df["entity_type"] == "CUSTOMER"].rename(columns={"entity_id": "customer_id", "severity": "anomaly_severity"})
    
    merged = pd.merge(r_df, a_df, on="customer_id", how="outer")
    merged = pd.merge(merged, n_df, on="customer_id", how="outer")
    merged["potential_revenue"] = 8500.0
    merged["win_probability"] = 0.75
    merged["expected_value"] = 6375.0
    return merged

def evaluate_condition(row: pd.Series, condition_expr: str) -> bool:
    """
    Safely evaluates a Python boolean expression against the row values.
    Transforms expressions like 'customer_risk.risk_score >= 70' into row variable lookups.
    """
    # Mapping prefixes from YAML rules to DataFrame column names
    expr = (
        condition_expr
        .replace("customer_risk.", "")
        .replace("anomaly_output.", "")
        .replace("sales_opportunity.", "")
        .replace("next_purchase_prediction.", "")
    )
    # Context dictionary for eval
    context = row.to_dict()
    try:
        return bool(eval(expr, {"__builtins__": {}}, context))
    except Exception:
        return False

def run_recommendations():
    logger.info("Starting DECIDE Layer Recommendation Engine...")
    rules = load_rules()
    data = fetch_predict_layer_consolidated()
    
    today = date.today()
    recommendations = []

    for _, row in data.iterrows():
        cust_id = row.get("customer_id")
        if not cust_id:
            continue

        for rule in rules:
            cond = rule.get("condition")
            if evaluate_condition(row, cond):
                # Render templates with actual context values
                ctx = {k: ("N/A" if pd.isna(v) else v) for k, v in row.to_dict().items()}
                
                # Format currency variables if present
                for num_key in ["revenue_at_risk", "deviation", "expected_purchase_value", "potential_revenue", "expected_value"]:
                    if num_key in ctx and isinstance(ctx[num_key], (int, float)):
                        ctx[num_key] = f"{ctx[num_key]:,.2f}"

                action = Template(rule.get("action_template", "")).safe_substitute(ctx)
                impact = Template(rule.get("impact_template", "")).safe_substitute(ctx)

                recommendations.append({
                    "customer_id": cust_id,
                    "as_of_date": today,
                    "issue_code": rule["issue_code"],
                    "issue_description": f"Triggered by rule: {rule['id']}",
                    "recommended_action": action,
                    "expected_impact": impact,
                    "priority": rule.get("priority", "MEDIUM"),
                    "status": "OPEN",
                    "action_metadata": "{}"
                })

    out_df = pd.DataFrame(recommendations)
    if not out_df.empty:
        # Drop duplicates in case multiple rules share issue_code
        out_df = out_df.drop_duplicates(subset=["customer_id", "issue_code", "as_of_date"])
        upsert_dataframe(out_df, "decision_recommendation", ["customer_id", "issue_code", "as_of_date"])
        logger.info(f"Upserted {len(out_df)} prescriptive recommendations into decision_recommendation.")
    return out_df

if __name__ == "__main__":
    run_recommendations()
