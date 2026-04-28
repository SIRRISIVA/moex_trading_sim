import numpy as np
from agents.base import BaseAgent

class RandomAgent(BaseAgent):
    def act(self, obs: dict) -> np.ndarray:
        w = np.random.randn(self.n_assets)
        w = w / (np.sum(np.abs(w)) + 1e-8)
        return w