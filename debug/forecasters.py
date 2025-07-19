from darts import TimeSeries
from darts.models import GlobalNaiveSeasonal, RegressionModel, AutoARIMA

from core import ForecastModel


class NaiveSeasonal(ForecastModel):
    def __init__(self, timesteps):
        self.model = GlobalNaiveSeasonal(input_chunk_length=timesteps, output_chunk_length=1)

    def fit(self, train: TimeSeries):
        self.model.fit(train)
        return self.model

    def predict(self, n: int) -> TimeSeries:
        forecast = self.model.predict(n)
        return forecast


class LinearRegression(ForecastModel):
    def __init__(self, timesteps):
        self.model = RegressionModel(lags=timesteps, output_chunk_length=1)

    def fit(self, train: TimeSeries):
        self.model.fit(train)
        return self.model

    def predict(self, n: int) -> TimeSeries:
        forecast = self.model.predict(n)
        return forecast


class ARIMA(ForecastModel):
    def __init__(self, p, d, q):
        self.model = AutoARIMA(p, d, q)

    def fit(self, train: TimeSeries):
        self.model.fit(train)
        return self.model

    def predict(self, n: int) -> TimeSeries:
        forecast = self.model.predict(n)
        return forecast
