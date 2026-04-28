import numpy as np

class BaseAgent:
    def __init__(self, n_assets: int):
        self.n_assets = n_assets

    def act(self, observation: dict) -> np.ndarray:
        raise NotImplementedError

    def reset(self):
        pass

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0)

        mean = np.mean(x)
        std = np.std(x)

        return (x - mean) / (std + 1e-8)

