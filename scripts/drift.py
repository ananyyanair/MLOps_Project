from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from loguru import logger
logger.add("logs/app.log", rotation="500 KB")

def detect_drift(train_data, new_data):
    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=train_data, current_data=new_data)
        report.save_html("artifacts/drift_report.html")
        logger.info("Drift report generated successfully")
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
