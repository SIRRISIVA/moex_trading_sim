import numpy as np
from agents.base import BaseAgent
from features.feature_config import FEATURE_INDEX

class MeanReversionAgent(BaseAgent):
    def act(self, obs):
        f = obs["features"]

        ret5 = f[:, FEATURE_INDEX["ret_5"]]

        score = -ret5
        score = self.normalize(score)

        w = np.zeros(self.n_assets)
        w[np.argmax(score)] = 1.0
        return w