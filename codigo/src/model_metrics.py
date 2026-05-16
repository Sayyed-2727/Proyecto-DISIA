from sklearn.metrics import precision_score, f1_score, roc_auc_score
from prometheus_client import Gauge

# Métricas Prometheus del modelo
MODEL_PRECISION = Gauge("model_precision", "Model precision score")
MODEL_F1 = Gauge("model_f1_score", "Model F1 score")
MODEL_AUC = Gauge("model_auc_score", "Model AUC score")

class ModelMetrics:

    @staticmethod
    def compute_and_export(y_true, y_pred_probs):
        """
        Calcula métricas de modelo y las exporta a Prometheus.
        """
        y_pred = [1 if p >= 0.5 else 0 for p in y_pred_probs]

        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_probs)

        MODEL_PRECISION.set(float(precision))
        MODEL_F1.set(float(f1))
        MODEL_AUC.set(float(auc))

        return {
            "precision": precision,
            "f1": f1,
            "auc": auc
        }
