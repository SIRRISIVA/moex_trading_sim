from agents.base import BaseAgent
import numpy as np

class LogisticRegressionAgent(BaseAgent):
    def __init__(self, n_assets, model):
        super().__init__(n_assets)
        self.model = model

    def act(self, obs):
        X = obs["features"]

        proba = self.model.predict_proba(X)[:, 1]

        k = 5
        idx = np.argsort(proba)[-k:]

        w = np.zeros(self.n_assets)
        w[idx] = 1 / k

        return w