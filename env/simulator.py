import numpy as np

class TradingSimulator:
    def __init__(
    self,
    n_assets: int,
    initial_cash: float = 1_000_000,
    commission: float = 0.0
    ):
        self.n_assets = n_assets
        self.initial_cash = initial_cash
        self.commission = commission

        self.reset()

    def reset(self):
        self.nav = self.initial_cash
        self.weights = np.zeros(self.n_assets)
        self.prev_prices = None

    def step(self, target_weights: np.ndarray, prices: np.ndarray):
        target_weights = target_weights / (np.sum(np.abs(target_weights)) + 1e-8)

        if self.prev_prices is not None:
            returns = (prices - self.prev_prices) / self.prev_prices
            portfolio_returns = np.dot(self.weights, returns)
            self.nav *= (1 + portfolio_returns)

        turnover = np.sum(np.abs(target_weights -  self.weights))
        cost = turnover * self.commission * self.nav
        self.nav -= cost

        self.weights = target_weights
        self.prev_prices = prices

        return self.nav