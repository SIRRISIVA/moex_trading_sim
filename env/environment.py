import numpy as np

class TradingEnvironment:
    def __init__(self, data_source, simulator):
        self.data = data_source
        self.sim = simulator

    def reset(self):
        self.data.reset()
        self.sim.reset()

        obs = self.data.get_observation()
        return obs

    def step(self, action: np.ndarray):
        prices = self.data.get_prices()
        prev_nav = self.sim.nav
        nav = self.sim.step(action, prices)
        reward = (nav - prev_nav) / (prev_nav + 1e-8)

        done = self.data.step_forward()

        obs = self.data.get_observation()

        info = {
            "nav": nav,
            "weights": self.sim.weights
        }

        return obs, reward, done, info