import mlrun
from mlrun.serving import V2ModelServer
import joblib
import pandas as pd


class TriageModel(V2ModelServer):
    """
    MLRun V2 compatible model server
    Compatible con Open Inference Protocol (V2)
    """

    def load(self):
        """Carga el modelo desde el artifact store"""
        model_file, _ = self.get_model(".pkl")
        self.model = joblib.load(model_file)

    def predict(self, body: dict):
        """
        Espera formato V2:
        {
            "inputs": [...]
        }
        """
        try:
            inputs = body["inputs"]
            df = pd.DataFrame(inputs)
            preds = self.model.predict(df)

            return {
                "predictions": preds.tolist()
            }

        except Exception as e:
            return {
                "error": str(e)
            }
