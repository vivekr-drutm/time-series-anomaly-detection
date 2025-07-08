from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# load nab data into a dictionary
def load_nab():
    nab_data_folder = Path('data/nab_data_corpus')
    nab_data = {}
    for dir in nab_data_folder.glob('*'):
        if (dir / dir.name).exists():
            nab_data[dir.name] = {}
            sub_dir = dir / dir.name
            for file in sub_dir.rglob('*.csv'):
                if file.is_file():
                    nab_data[dir.name][file.name[:-4]] = {}
                    nab_data[dir.name][file.name[:-4]]['path'] = file
                    df = pd.read_csv(file)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    nab_data[dir.name][file.name[:-4]]['df'] = df
    return nab_data

def plot_folder_files(folder):
    if 'nab_data' not in globals():
        nab_data = load_nab()
    folder_data = nab_data[folder]
    fig, axes = plt.subplots(len(folder_data.keys()), 1, figsize=(18,12))
    axes.flatten()
    for index, file in enumerate(nab_data[folder].keys()):
        nab_data[folder][file]['df'].plot(x='timestamp', y='value', ax=axes[index], title=file, legend=False)
    plt.tight_layout(pad=2.0)


class ThresholdConfig:
    def __init__(self, input_series: pd.DataFrame):
        if 'value' not in input_series.columns:
            self.series_df = input_series.set_axis(['value'], axis=1, copy=True)
        else:
            self.series_df = input_series

        self.outliers = pd.DataFrame()
        self.anomaly_score_df = pd.DataFrame()
        self.built_from_anom_scores = False

    @classmethod
    def from_anomaly_score(cls, anomaly_scores: pd.DataFrame, input_series: pd.DataFrame):
        anom_score_threshold = cls(input_series)
        anom_score_threshold.built_from_anom_scores = True
        anom_score_threshold.anomaly_score_df = anomaly_scores
        return anom_score_threshold


    def static(self, threshold, anomaly_score: pd.DataFrame = None, target_series: pd.DataFrame = None):
        if not self.built_from_anom_scores:

            if 'value' not in anomaly_score.columns:
                self.anomaly_score_df = anomaly_score.set_axis(['value'], axis=1, copy=True)
            else:
                self.anomaly_score_df = anomaly_score

        if target_series is None:
           target_series = self.series_df

        outliers_mask = self.anomaly_score_df['value'] > threshold
        if outliers_mask.size != target_series.size:
            length = target_series.size

            reduced_length_series = target_series[length-outliers_mask.size:]
            self.outliers = reduced_length_series[outliers_mask].copy()
            return self.outliers
        else:
            self.outliers = target_series[outliers_mask].copy()

            return self.outliers

    def static_on_score(self, threshold):
        if self.built_from_anom_scores:
            return self.static(threshold, self.anomaly_score_df, self.anomaly_score_df)
        else:
            raise Exception('Must use ThresholdConfig.from_anomaly_score() to use thresholds on anomaly scores')

    def rolling_window(self, window, threshold, target_series: pd.DataFrame = None):
        if target_series is None:
            target_series = self.series_df

        threshold_values = target_series.rolling(window).quantile(threshold)
        outliers_mask = target_series['value'] >= threshold_values['value']
        self.outliers = target_series[outliers_mask].dropna().copy()
        # print(target_series[outliers_mask].dropna())
        return self.outliers

    def rolling_window_on_score(self, window, threshold):
        if self.built_from_anom_scores:
            return self.rolling_window(window, threshold, self.anomaly_score_df)
        else:
            raise Exception('Must use ThresholdConfig.from_anomaly_score() to use thresholds on anomaly scores')

    def z_score(self, window, threshold, target_series: pd.DataFrame = None):
        if target_series is None:
            target_series = self.series_df

        rolling_win = target_series.rolling(window)
        z_scores = ((target_series - rolling_win.mean().fillna(0)) / rolling_win.std().fillna(1)).iloc[window:]
        outliers_mask = abs(z_scores['value']) > threshold
        outlier_indices = z_scores[outliers_mask].index
        self.outliers = target_series.loc[outlier_indices].copy()
        # print(((target_series - rolling_win.mean().fillna(0))/rolling_win.std().fillna(1))[window:])
        return self.outliers

    def z_score_on_score(self, window, threshold):
        if self.built_from_anom_scores:
            return self.z_score(window, threshold, self.anomaly_score_df)
        else:
            raise Exception('Must use ThresholdConfig.from_anomaly_score() to use thresholds on anomaly scores')

    def plot(self):
        series_outlier_values = self.series_df.loc[self.outliers.index, 'value']
        plt.plot(self.outliers.index, series_outlier_values, '.', color='r')
        return pd.DataFrame(index=self.outliers.index, data={'value': series_outlier_values}, copy=True)
