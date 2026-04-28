import pandas as pd
import numpy as np

from features.alpha_factors import (
    ret_1, ret_5, ret_20,
    vol_10, zscore_20
)

def rolling_zscore(df: pd.Series, window=25) -> pd.Series:
    mean = df.rolling(window).mean()
    std = df.rolling(window).std()
    return (df - mean) / (std + 1e-8)

def winsorize(df: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    # Считаем границы только на основе уже увиденных данных
    lower_bound = df.expanding().quantile(lower)
    upper_bound = df.expanding().quantile(upper)
    return df.clip(lower_bound, upper_bound, axis=0)


def build_features(prices: pd.DataFrame) -> pd.DataFrame:

    f_ret1 = ret_1(prices)
    f_ret5 = ret_5(prices)
    f_ret20 = ret_20(prices)
    f_vol_10 = vol_10(prices)
    f_zscore_20 = zscore_20(prices)

    f_ret1 = winsorize(f_ret1)
    f_ret5 = winsorize(f_ret5)
    f_ret20 = winsorize(f_ret20)

    f_ret5 = rolling_zscore(f_ret5, 20)
    f_ret20 = rolling_zscore(f_ret20, 20)
    f_vol_10 = rolling_zscore(f_vol_10, 20)

    features: pd.DataFrame = pd.concat({
        'ret_1': f_ret1,
        'ret_5': f_ret5,
        'ret_20': f_ret20,
        'vol_10': f_vol_10,
        'zscore_20': f_zscore_20
    }, axis=1)

    features = features.shift(1)
    # Удаляем строки, где индикаторы еще не успели рассчитаться (NaN)
    features = features.dropna()


    return features