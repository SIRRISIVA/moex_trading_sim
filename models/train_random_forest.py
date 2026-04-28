import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit


def train_rf(features, close):
    returns = close.pct_change()
    target = (returns > 0).astype(int)

    X = features.stack(future_stack=True).values
    y = target.stack(future_stack=True).values

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    tscv = TimeSeriesSplit(n_splits=5)
    scores= []
    for train_idx, val_idx in tscv.split(X):
        model = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=5, min_samples_split=20)
        model.fit(X[train_idx], y[train_idx])
        score = model.score(X[val_idx], y[val_idx])
        scores.append(score)

    print("CV accuracy:", np.mean(scores))

    model = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=5, min_samples_split=20)
    model.fit(X, y)

    return model