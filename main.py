from agents.ml.logistic_regression import LogisticRegressionAgent
from env.data_source import DataSource
from env.simulator import TradingSimulator
from env.environment import TradingEnvironment

from agents.baseline.random import RandomAgent
from agents.baseline.equal_weight import EqualWeightAgent
from agents.baseline.momentum import MomentumAgent
from agents.baseline.mean_reversion import MeanReversionAgent
from agents.baseline.volatility_target import VolatilityTargetAgent
from agents.ml.logistic_regression import LogisticRegressionAgent
from agents.ml.random_forest import RandomForestAgent

from models.train_logistic_regression import train_model
from models.train_random_forest import train_rf

from evaluation.runner import compare

# --------------------
# INIT
# --------------------
data = DataSource("moex_data_48.h5")
sim = TradingSimulator(n_assets=data.n_assets)
env = TradingEnvironment(data, sim)

split_date = "2025-01-01"

train_features = data.features.loc[:split_date]
train_close = data.close.loc[:split_date]

test_features = data.features.loc[split_date:]
test_close = data.close.loc[split_date:]

# MODELS TRAINING
model_lr = train_model(train_features, train_close)
model_rf = train_rf(train_features, train_close)

# --------------------
# AGENTS
# --------------------
agents = {
    "random": RandomAgent(data.n_assets),
    "equal": EqualWeightAgent(data.n_assets),
    "momentum": MomentumAgent(data.n_assets),
    "mean_reversion": MeanReversionAgent(data.n_assets),
    "volatility_target": VolatilityTargetAgent(data.n_assets),
    "logreg": LogisticRegressionAgent(data.n_assets, model_lr),
    "rf": RandomForestAgent(data.n_assets, model_rf)
}

# --------------------
# RUN
# --------------------
data.features = test_features
data.close = test_close
data.dates = test_close.index

env = TradingEnvironment(data, sim)

results = compare(env, agents)