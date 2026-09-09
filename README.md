# InsightEdge: PREDICT & DECIDE Layer

This project adds a predictive and prescriptive analytics layer on top of your existing Power BI sales dashboard.

---

## Quick Start (Docker Compose - Single Command)

To spin up **PostgreSQL** (with schema auto-applied), the **FastAPI What-If Simulator**, and the **Prefect Pipeline Worker**:

```bash
docker-compose up --build -d
```

Check running container health:
```bash
docker-compose ps
```

---

## Local Development & Testing

### 1. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Test Suite

Run the pytest suite to verify the FastAPI simulator endpoint and YAML rules engine:

```bash
pytest tests/ -v
```

### 3. Start PostgreSQL Database

Ensure PostgreSQL is running locally or via Docker:
```bash
docker-compose up -d postgres
```

Apply the SQL schema:
```bash
PGPASSWORD=postgrespassword psql -h localhost -U postgres -d sales_insights -f sql/schema.sql
```

### 4. Run Machine Learning Jobs Manually

Each job can run independently:

```bash
# Sales forecasting (Prophet)
python -m src.jobs.forecast_job

# Customer risk scoring (XGBoost + SHAP explainability)
python -m src.jobs.risk_job

# Anomaly detection (Rolling Z-Score)
python -m src.jobs.anomaly_job

# Next purchase interval prediction
python -m src.jobs.next_purchase_job

# DECIDE layer rules-based recommendation engine
python -m src.jobs.recommendation_job
```

### 5. Run the Full Prefect Pipeline

Execute the end-to-end daily orchestration flow:

```bash
python -m src.flows.daily_pipeline
```

### 6. Run the FastAPI What-If Simulator

Start the API locally:
```bash
uvicorn src.api.main:app --reload --port 8000
```

Access Interactive Swagger Documentation:
- Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

Test the Simulation Endpoint with cURL:
```bash
curl -X POST "http://localhost:8000/api/v1/simulate/forecast" \
     -H "Content-Type: application/json" \
     -d '{
       "scenario_name": "Recover High Risk Accounts",
       "recovered_customer_ids": ["CUST_0001", "CUST_0002", "CUST_0003"],
       "uplift_percentage": 25.0,
       "horizon_months": 6
     }'
```

---

## Power BI Integration

1. In Power BI, connect to PostgreSQL:
   - Server: `localhost:5432` (or your Postgres host)
   - Database: `sales_insights`
2. Load the 6 newly populated tables:
   - `sales_forecast`
   - `customer_risk`
   - `sales_opportunity`
   - `anomaly_output`
   - `decision_recommendation`
   - `next_purchase_prediction`
3. Join these tables to your existing `Customer` and `Date` dimensions using `customer_id` and date keys.
