"""
FastAPI Simulation & Control Center UI Service.
Supports Power BI 'What-If' Parameter Controls and provides interactive dashboard for testing PREDICT & DECIDE layers.
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from src.jobs.recommendation_job import run_recommendations
from src.jobs.risk_job import run_risk_scoring
from src.jobs.anomaly_job import run_anomaly_detection
from src.jobs.next_purchase_job import run_next_purchase_prediction

app = FastAPI(
    title="InsightEdge What-If Simulator API",
    description="Power BI interactive scenario simulator for real-time forecast adjustment",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScenarioRequest(BaseModel):
    scenario_name: str = Field(default="Customer Recovery Scenario", json_schema_extra={"example": "Q3 Churn Recovery"})
    recovered_customer_ids: List[str] = Field(
        default=["CUST_0001", "CUST_0003"],
        json_schema_extra={"example": ["CUST_0001", "CUST_0002"]}
    )
    uplift_percentage: float = Field(
        default=25.0,
        description="Expected recovery uplift % in sales",
        json_schema_extra={"example": 20.0}
    )
    horizon_months: int = Field(default=6, ge=1, le=24, json_schema_extra={"example": 6})

class ForecastDataPoint(BaseModel):
    month: str
    baseline_forecast: float
    scenario_forecast: float
    simulated_delta: float

class ScenarioResponse(BaseModel):
    scenario_name: str
    target_customers: List[str]
    total_baseline_sales: float
    total_simulated_sales: float
    net_incremental_revenue: float
    monthly_breakdown: List[ForecastDataPoint]

STATIC_DIR = os.path.join(os.path.dirname(__file__), "../static")

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h2>InsightEdge UI is starting...</h2>")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "what-if-simulator"}

@app.get("/api/v1/recommendations")
def get_recommendations():
    """Fetches or calculates DECIDE prescriptive recommendations."""
    df = run_recommendations()
    return df.to_dict(orient="records")

@app.get("/api/v1/risk")
def get_risk_scores():
    """Fetches or calculates PREDICT customer risk & SHAP metrics."""
    df = run_risk_scoring()
    return df.to_dict(orient="records")

@app.get("/api/v1/anomalies")
def get_anomalies():
    """Fetches or calculates PREDICT rolling Z-Score sales anomalies."""
    df = run_anomaly_detection()
    return df.to_dict(orient="records")

@app.get("/api/v1/next-purchases")
def get_next_purchases():
    """Fetches or calculates PREDICT next-purchase interval forecasts."""
    df = run_next_purchase_prediction()
    return df.to_dict(orient="records")

@app.post("/api/v1/simulate/forecast", response_model=ScenarioResponse)
def simulate_forecast_scenario(request: ScenarioRequest):
    """
    Simulates recalculated sales forecast if specific at-risk customers are recovered
    or given an uplift percentage.
    """
    if not request.recovered_customer_ids:
        raise HTTPException(status_code=400, detail="Must provide at least one customer ID.")

    dates = pd.date_range(start=pd.Timestamp.today().replace(day=1), periods=request.horizon_months, freq='MS')
    monthly_data = []

    total_baseline = 0.0
    total_simulated = 0.0

    # Base simulation calculation
    base_per_cust_monthly = 4200.0  # Normalized base per account
    num_cust = len(request.recovered_customer_ids)

    for d in dates:
        month_str = d.strftime("%Y-%m")
        # Baseline sales for selected customers
        baseline_month = round(num_cust * base_per_cust_monthly * (1.0 + np.random.uniform(-0.05, 0.05)), 2)
        # Uplift applied to recovered customers
        multiplier = 1.0 + (request.uplift_percentage / 100.0)
        simulated_month = round(baseline_month * multiplier, 2)
        delta = round(simulated_month - baseline_month, 2)

        total_baseline += baseline_month
        total_simulated += simulated_month

        monthly_data.append(ForecastDataPoint(
            month=month_str,
            baseline_forecast=baseline_month,
            scenario_forecast=simulated_month,
            simulated_delta=delta
        ))

    return ScenarioResponse(
        scenario_name=request.scenario_name,
        target_customers=request.recovered_customer_ids,
        total_baseline_sales=round(total_baseline, 2),
        total_simulated_sales=round(total_simulated, 2),
        net_incremental_revenue=round(total_simulated - total_baseline, 2),
        monthly_breakdown=monthly_data
    )
