import numpy as np
import pandas as pd


def synthetic_returns(
    prices: pd.DataFrame, information_ratio: float, seed: int = 0
) -> pd.DataFrame:
    """
    prices: a DataFrame of prices
    information_ratio: the desired information ratio of the synthetic returns

    returns: a DataFrame of "synthetic return predictions" computed as
    alpha*(returns+noise), where alpha=var_r / (var_r + var_eps); this is the
    coefficient that minimize the variance of the prediction error under the
    above model.
    """
    np.random.seed(seed)

    returns = prices.pct_change()
    var_r = returns.var()

    alpha = information_ratio**2
    var_eps = var_r * (1 - alpha) / alpha
    noise = np.random.normal(0, np.sqrt(var_eps), size=returns.shape)
    synthetic_returns = alpha * (returns + noise)
    return synthetic_returns
