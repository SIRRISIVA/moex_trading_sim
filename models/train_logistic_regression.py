from sklearn.linear_model import LogisticRegression
import numpy as np

def train_model(features, close):
    returns = close.pct_change().shift(-1)
    target = (returns > 0).astype(int)

    X = features.stack().values
    y = target.stack().values

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model