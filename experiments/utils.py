import numpy as np
import pandas as pd


def synthetic_returns(
    prices: pd.DataFrame,
    information_ratio: float,
    smoothing_len: int = 21,
    seed: int = 0,
) -> pd.DataFrame:
    """
    prices: a DataFrame of prices
    information_ratio: the desired information ratio of the synthetic returns
    smoothing_len: the length of the smoothing window for the synthetic returns
    seed: random seed for reproducibility

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

    # Smoothe the synthetic returns

    returns_smoothed = synthetic_returns.shift(-smoothing_len + 1)
    returns_smoothed = pd.concat(
        [synthetic_returns.iloc[: smoothing_len - 1], returns_smoothed]
    )
    # update first 4 index to be 0,1,2,3
    index_new = returns_smoothed.index.tolist()
    index_new[: smoothing_len - 1] = [i for i in range(smoothing_len - 1)]
    returns_smoothed.index = index_new

    return returns_smoothed.ffill().dropna()


def initialize_markowitz():
    pass
