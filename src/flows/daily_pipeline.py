"""
Prefect 2/3 Orchestration Pipeline for PREDICT & DECIDE layers.
Schedules jobs daily, enforces dependency graph, and alerts on failure.
"""
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
import requests
import logging
from src.common.config import settings
from src.jobs.forecast_job import run_sales_forecasting
from src.jobs.risk_job import run_risk_scoring
from src.jobs.anomaly_job import run_anomaly_detection
from src.jobs.next_purchase_job import run_next_purchase_prediction
from src.jobs.recommendation_job import run_recommendations

logger = logging.getLogger(__name__)

def send_slack_alert(message: str):
    if settings.slack_webhook_url:
        try:
            requests.post(settings.slack_webhook_url, json={"text": message}, timeout=5)
        except Exception as e:
            logger.error(f"Failed to deliver Slack alert: {e}")
    else:
        logger.warning(f"Slack webhook not configured. Alert suppressed: {message}")

@task(name="Forecast Sales Task", retries=2, retry_delay_seconds=30)
def task_sales_forecasting():
    logger.info("Executing Prophet Sales Forecast...")
    return run_sales_forecasting(horizon_months=6)

@task(name="Customer Risk Task", retries=2, retry_delay_seconds=30)
def task_customer_risk():
    logger.info("Executing XGBoost Risk & SHAP Scoring...")
    return run_risk_scoring()

@task(name="Anomaly Detection Task", retries=2, retry_delay_seconds=30)
def task_anomaly_detection():
    logger.info("Executing Rolling Z-Score Anomaly Detection...")
    return run_anomaly_detection(z_threshold=2.0)

@task(name="Next Purchase Task", retries=2, retry_delay_seconds=30)
def task_next_purchase():
    logger.info("Executing Interval Next-Purchase Prediction...")
    return run_next_purchase_prediction()

@task(name="Recommendation Engine Task", retries=1, retry_delay_seconds=15)
def task_decision_recommendations():
    logger.info("Executing Rule-based Decision Recommendation Engine...")
    return run_recommendations()

@flow(
    name="predict_decide_daily_pipeline",
    task_runner=SequentialTaskRunner(),
    description="Daily pipeline orchestrating PREDICT model jobs and DECIDE rules engine."
)
def predict_decide_daily_pipeline():
    try:
        # Phase 1: Run PREDICT layer model jobs in parallel or sequence
        forecast_res = task_sales_forecasting()
        risk_res = task_customer_risk()
        anomaly_res = task_anomaly_detection()
        np_res = task_next_purchase()

        # Phase 2: DECIDE layer (Prescriptive Recommendations) depends on PREDICT outputs
        rec_res = task_decision_recommendations(wait_for=[risk_res, anomaly_res, np_res])

        logger.info("Pipeline executed successfully.")
        return {"status": "SUCCESS"}
    except Exception as exc:
        err_msg = f"❌ *PREDICT & DECIDE Daily Pipeline Failed*:\n```{str(exc)}```"
        send_slack_alert(err_msg)
        raise exc

if __name__ == "__main__":
    predict_decide_daily_pipeline()
