import pandas as pd
import numpy as np


def ret_1(prices):
    return prices.pct_change()


def ret_5(prices):
    return prices.pct_change(5)


def ret_20(prices):
    return prices.pct_change(20)


def vol_10(prices):
    return prices.pct_change().rolling(10).std()


def zscore_20(prices):
    mean = prices.rolling(20).mean()
    std = prices.rolling(20).std()
    return (prices - mean) / (std + 1e-8)