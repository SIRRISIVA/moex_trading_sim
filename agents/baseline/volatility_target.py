import numpy as np
from agents.base import BaseAgent
from features.feature_config import FEATURE_INDEX

class VolatilityTargetAgent(BaseAgent):
    def act(self, obs):
        f = obs["features"]

        vol = f[:, FEATURE_INDEX["vol_10"]]

        inv_vol = 1.0 / (np.abs(vol) + 1e-6)
        inv_vol = self.normalize(inv_vol)

        w = np.maximum(inv_vol, 0)
        return w / (np.sum(w) + 1e-8)