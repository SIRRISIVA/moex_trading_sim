import pandas as pd
import numpy as np

from features.feature_pipeline import build_features


class DataSource:
    def __init__(self, path: str):
        df = pd.read_hdf(path, 'moex/prices')

        df = df[
            (df['NUMTRADES'] > 0) &
            (df['VOLUME'] > 0) &
            (df['BOARDID'] == 'TQBR')
        ]

        df = df.groupby(['date', 'ticker']).last()
        close = df['CLOSE'].unstack('ticker')
        close = close.ffill().dropna()

        features = build_features(close)
        close = close.loc[features.index]

        self.close = close
        self.features = features
        self.dates = close.index
        self.tickers = close.columns
        self.n_assets = len(self.tickers)
        self.n_features = len(features.columns.levels[0])

        self.step = 0

    def reset(self):
        self.step = 0

    def get_observation(self):

        date = self.dates[self.step]
        row = self.features.loc[date]
        obs = row.values.reshape(self.n_features, self.n_assets).T

        return {
            'features': obs,
            'prices': self.close.loc[date].values,
            'tickers': self.tickers,
            'date': date
        }

    def step_forward(self):
        self.step += 1
        return self.step >= len(self.dates) - 1

    def get_prices(self):
        date = self.dates[self.step]
        return self.close.loc[date].values