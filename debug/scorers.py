from darts.ad import NormScorer

from core import ScorerModel


class EuclideanScorer(ScorerModel):
    def __init__(self):
        self.scorer = NormScorer(ord=2)
        self.anomaly_scores = None

    def score(self, test, forecast):
        self.anomaly_scores = self.scorer.score_from_prediction(test, forecast)
        return self.anomaly_scores

