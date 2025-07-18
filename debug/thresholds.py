import pandas as pd

from core import ThresholdModel


class StaticThreshold(ThresholdModel):
    def __init__(self, threshold, target_series):
        self.threshold = threshold
        if isinstance(target_series, pd.DataFrame):
            self.target_series = target_series
        else:
            self.target_series = target_series.to_frame()
        self.anomaly_score_df = pd.DataFrame()
        return

    def identify_outliers(self, anomaly_score: pd.DataFrame = None):
        if anomaly_score is None:
            raise ValueError("anomaly_score cannot be None")
        # Ensure anomaly_score is a DataFrame with 'value' column
        if isinstance(anomaly_score, pd.Series):
            self.anomaly_score_df = anomaly_score.to_frame(name='value')
        elif 'value' not in anomaly_score.columns:
            self.anomaly_score_df = anomaly_score.set_axis(['value'], axis=1, copy=True)
        else:
            self.anomaly_score_df = anomaly_score.copy()

        outliers_mask = self.anomaly_score_df['value'] > self.threshold

        if outliers_mask.size != self.target_series.size:
            # Align by index instead of position
            common_index = self.anomaly_score_df.index.intersection(self.target_series.index)
            aligned_mask = outliers_mask.loc[common_index]
            return self.target_series.loc[common_index][aligned_mask].copy()
        else:
            return self.target_series[outliers_mask].copy()


class RollingWindowThreshold(ThresholdModel):
    def __init__(self, threshold, target_series):
        self.threshold = threshold
        self.target_series = target_series
        self.outliers = pd.DataFrame()
        return

    def identify_outliers(self, window):
        threshold_values = self.target_series.rolling(window).quantile(self.threshold)
        outliers_mask = self.target_series['value'] >= threshold_values['value']
        self.outliers = self.target_series[outliers_mask].dropna().copy()
        return self.outliers

class ZScoreThreshold(ThresholdModel):
    def __init__(self, threshold, target_series):
        self.threshold = threshold
        self.target_series = target_series

    def identify_outliers(self, window):
        rolling_win = self.target_series.rolling(window)
        z_scores = ((self.target_series - rolling_win.mean().fillna(0)) / rolling_win.std().fillna(1)).iloc[window:]
        outliers_mask = abs(z_scores['value']) > self.threshold
        outlier_indices = z_scores[outliers_mask].index
        return self.target_series.loc[outlier_indices].copy()
