import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.jobs.recommendation_job import load_rules, evaluate_condition
import pandas as pd

client = TestClient(app)

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_simulate_forecast_endpoint():
    payload = {
        "scenario_name": "Test Recovery",
        "recovered_customer_ids": ["CUST_0001", "CUST_0002"],
        "uplift_percentage": 20.0,
        "horizon_months": 3
    }
    response = client.post("/api/v1/simulate/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["scenario_name"] == "Test Recovery"
    assert len(data["monthly_breakdown"]) == 3
    assert data["total_simulated_sales"] > data["total_baseline_sales"]

def test_recommendation_rules_eval():
    rules = load_rules("config/recommendation_rules.yaml")
    assert len(rules) > 0

    sample_row = pd.Series({
        "risk_score": 85.0,
        "revenue_at_risk": 15000.0,
        "anomaly_severity": "CRITICAL",
        "anomaly_type": "DROP",
        "deviation": -4500.0,
        "days_until_next_purchase": -5,
        "win_probability": 0.8,
        "potential_revenue": 10000.0
    })

    # Test churn rule condition
    churn_rule = next(r for r in rules if r["id"] == "RULE_CHURN_PREVENTION")
    assert evaluate_condition(sample_row, churn_rule["condition"]) is True
