from typing import Dict

import pandas as pd

from core import ForecastModel, ScorerModel, ThresholdModel


class NABAnomalyDetector:
    def __init__(self, forecaster: ForecastModel, scorer: ScorerModel, threshold: ThresholdModel):
        self.forecast_model = forecaster
        self.anomaly_score_model = scorer
        self.threshold = threshold

    def detect(self, n, test) -> Dict[str, pd.DataFrame]:
        forecast = self.forecast_model.predict(n)
        anomaly_scores = self.anomaly_score_model.score(test, forecast)

        if hasattr(anomaly_scores, 'to_dataframe'):
            anomaly_scores_df = anomaly_scores.to_dataframe()
        else:
            anomaly_scores_df = anomaly_scores

        if 'value' not in anomaly_scores_df.columns:
           anomaly_scores_df.columns = ['value']

        if hasattr(self.threshold, 'identify_outliers'):
            if self.threshold.__class__.__name__ in ['RollingWindowThreshold', 'ZScoreThreshold']:
                anomalies = self.threshold.identify_outliers(n)
            else:
                anomalies  = self.threshold.identify_outliers(anomaly_scores_df)
        else:
            raise AttributeError("Threshold model must implement identify_outliers method")

        if hasattr(forecast, 'to_dataframe'):
            forecast_df = forecast.to_dataframe()
        else:
            forecast_df = forecast

        return {'forecast':forecast_df, 'anomaly_scores':anomaly_scores_df, 'anomalies':anomalies}