import numpy as np
from agents.base import BaseAgent

class EqualWeightAgent(BaseAgent):
    def act(self, obs: dict) -> np.ndarray:
        return np.ones(self.n_assets) / self.n_assets