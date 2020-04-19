from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class CrossValidateETS():

    def __init__(self, model, data, cv):
        self.data = data.copy()
        self.cv = cv
        self.model = model
    
    def _get_model(self, data):
        return self.model(data).fit(optimized=True)

    def _compute_scores(self, y_true, y_pred):
        return {"mae": mean_absolute_error(y_true, y_pred),
                "mse": mean_squared_error(y_true, y_pred)}

    def cross_validate(self):

        scores = {}

        scores.setdefault("mse", [])
        scores.setdefault("mae", [])

        for train_idx, test_idx in self.cv.split(self.data):
            model = self._get_model(self.data[train_idx])
            preds = model.forecast(len(test_idx))

            scores_h = self._compute_scores(self.data[test_idx], preds)

            scores.setdefault("mse", []).append(scores_h["mse"])
            scores.setdefault("mae", []).append(scores_h["mae"])

        return scores

        