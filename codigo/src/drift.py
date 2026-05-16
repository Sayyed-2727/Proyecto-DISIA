import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from prometheus_client import Gauge

# Métrica Prometheus para drift
DATA_DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Data drift score between reference and current data"
)

class DriftDetector:

    def __init__(self, reference_data_path: str):
        self.reference_data = pd.read_csv(reference_data_path)

    def compute_drift(self, current_data: pd.DataFrame):
        """
        Calcula drift entre datos de referencia y datos actuales.
        """
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )

        result = report.as_dict()

        drift_score = result["metrics"][0]["result"]["dataset_drift"]

        # Exportamos como métrica Prometheus
        DATA_DRIFT_SCORE.set(float(drift_score))

        return drift_score
