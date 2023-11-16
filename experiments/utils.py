import numpy as np
import pandas as pd


def synthetic_returns(
    prices: pd.DataFrame, information_ratio: float, forward_smoothing: int
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
    rng = np.random.default_rng(1)

    returns = prices.pct_change()
    returns = returns.rolling(forward_smoothing).mean().shift(-(forward_smoothing - 1))
    var_r = returns.var()

    alpha = information_ratio**2
    var_eps = var_r * (1 - alpha) / alpha
    noise = rng.normal(0, np.sqrt(var_eps), size=returns.shape)
    synthetic_returns = alpha * (returns + noise)

    return synthetic_returns


if __name__ == "__main__":
    prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    synthetic_returns = synthetic_returns(
        prices, information_ratio=0.07, forward_smoothing=21
    )
    returns = prices.pct_change()
    print(
        (
            np.sign(synthetic_returns - returns.mean())
            == np.sign(returns - returns.mean())
        )
        .mean(axis=0)
        .describe()
    )
