from abc import ABC, abstractmethod

from darts import TimeSeries


# I want these models:
# ARIMA
# NaiveSeasonal
# model = GlobalNaiveSeasonal(input_chunk_length=24, output_chunk_length=1)
# model.fit(train)

class ForecastModel(ABC):

    @abstractmethod
    def fit(self, data: TimeSeries):
        pass

    @abstractmethod
    def predict(self, n: int) -> TimeSeries:
        pass

class ScorerModel(ABC):
    @abstractmethod
    def score(self, test, forecast):
        pass

class ThresholdModel(ABC):
    @abstractmethod
    def identify_outliers(self):
        pass