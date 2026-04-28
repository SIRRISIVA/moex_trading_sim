import numpy as np
from agents.base import BaseAgent
from features.feature_config import FEATURE_INDEX

class MomentumAgent(BaseAgent):
    def __init__(self, n_assets: int):
        super().__init__(n_assets)
        self.w1 = 0.7
        self.w2 = 0.3

    def act(self, obs):
        f = obs["features"]

        ret5 = f[:, FEATURE_INDEX["ret_5"]]
        ret20 = f[:, FEATURE_INDEX["ret_20"]]

        score = self.w1 * ret5 + self.w2 * ret20
        score = self.normalize(score)

        w = np.zeros(self.n_assets)
        w[np.argmax(score)] = 1.0

        return w