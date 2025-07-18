import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.ad import NormScorer
from darts.models import RegressionModel

from nab_utils import load_nab, ThresholdConfig


def distort_time_series(df, anoms=['zero'], return_anomalies=False, max_length=108):
    distorted = df.copy()
    original_values = distorted['value']
    distort_length = np.random.randint(1, max_length)

    distorted_values = original_values * 0.95 + original_values.std()*0.1*np.random.choice([1, -1])
    distorted['value'] = distorted_values

    all_anomalies = distorted.timestamp.to_frame()
    all_anomalies['value'] = 0
    if type(anoms) is str:
        anom_key = anoms
        anoms = []
        anoms.append(anom_key)
    for anom_id in range(len(anoms)):
        start_idx = np.random.randint(int(0.7*distorted['timestamp'].size), distorted['timestamp'].size-distort_length)

        anomaly_types = {'zero': 0, 'minor_drop': 0.95, 'moderate_drop': 0.7, 'severe_drop': 0.35, 'critical_drop': 0.15}

        anomaly_values = distorted.loc[start_idx:start_idx+distort_length, "value"] * anomaly_types[anoms[anom_id]]
        distorted.loc[start_idx:start_idx+distort_length, "value"] = anomaly_values


        if return_anomalies:
            all_anomalies.loc[start_idx:start_idx+distort_length, 'value'] = 1
    if not return_anomalies:
        return distorted

    # returns dataframes
    return distorted, all_anomalies


def score_anomalies(time_series1: pd.DataFrame, time_series2: pd.DataFrame,
                    model, predict_time_steps,
                    ret_forecast=False):
    split_val = 0.8
    difference = ((time_series2 - time_series1) / time_series2) * 100
    difference_ts = TimeSeries.from_dataframe(difference)

    train, test = TimeSeries.split_after(difference_ts, split_val)
    model.fit(train, )
    forecast = model.predict(predict_time_steps)

    scorer = NormScorer(ord=2)
    anomaly_scores = scorer.score_from_prediction(test, forecast)
    if ret_forecast:
        return anomaly_scores.to_dataframe(), forecast.to_dataframe()
    return anomaly_scores.to_dataframe()

if __name__ == '__main__':
    # nab_data = load_nab()

    # print('all nab data loaded')

    # create the second time series from selected NAB series
    # original_df = nab_data['artificialWithAnomaly']['art_daily_jumpsdown']['df']

    original_df = pd.read_csv("data/nab_data_corpus/artificialWithAnomaly/artificialWithAnomaly/art_daily_jumpsdown.csv")
    print("data loaded")

    # randomly choose anomaly types
    anomaly_types = ['zero', 'minor_drop', 'moderate_drop', 'severe_drop', 'critical_drop']
    insert_anomalies = []
    num_anomalies = 3
    for i in range(num_anomalies):
        insert_anomalies.append(anomaly_types[np.random.randint(num_anomalies)])

    # create second time series with anomalies
    distorted_df, known_anomalies = distort_time_series(original_df, anoms=anomaly_types, return_anomalies=True)
    print("created second time series")

    # resample the data based on inputs
    original_df['timestamp'] = pd.to_datetime(original_df['timestamp'])
    distorted_df['timestamp'] = pd.to_datetime(distorted_df['timestamp'])
    original_hourly_df = original_df.set_index('timestamp').resample('h').mean()
    distorted_hourly_df = distorted_df.set_index('timestamp').resample('h').mean()

    model = RegressionModel(lags=24, output_chunk_length=1)
    anom_scores_df, forecast = score_anomalies(distorted_hourly_df, original_hourly_df,
                                     model, 66,
                                     ret_forecast=True)

    print("scored anomalies")

    threshold = ThresholdConfig.from_anomaly_score()
    # distorted_hourly_df.plot(ax=ax1)
    # print("creating anomaly plots")
    # forecast.plot(ax=ax2, color='g')
    # anom_scores_df.plot(ax=ax2, color='r')

    plt.show()







