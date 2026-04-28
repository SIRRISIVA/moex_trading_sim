import pandas as pd

class MOEXLoader:
    def __init__(self):
        pass

    def load_hdf(self, path):
        df = pd.read_hdf(path, 'moex/prices')
        return self._clean(df)

    def _clean(self, df):
        df = df[
            (df['NUMTRADES'] > 0) &
            (df['VOLUME'] > 0) &
            (df['BOARDID'] == 'TQBR')
        ]

        df = df.groupby(['date', 'ticker']).last()
        df = df.sort_index()

        return df

    def to_prices(selfself, df):
        prices = df['CLOSE'].unstack('ticker')
        prices = prices.ffill().dropna()
        return prices