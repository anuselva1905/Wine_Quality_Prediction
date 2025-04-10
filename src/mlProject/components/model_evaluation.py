import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import numpy as np
import joblib
from pathlib import Path
import json
from mlProject.utils.common import save_json
from mlProject.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def save_results(self):
        # Load test data and trained model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        # Prepare test features and target
        test_x = test_data.drop(columns=[self.config.target_column])
        test_y = test_data[self.config.target_column]

        # Make predictions
        predicted_qualities = model.predict(test_x)

        # Calculate evaluation metrics
        rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)

        # Save results as JSON
        scores = {"rmse": rmse, "mae": mae, "r2": r2}
        with open(self.config.metric_file_path, "w") as f:  # ✅ Fixed attribute name
            json.dump(scores, f, indent=4)

        print("Evaluation metrics saved successfully.")