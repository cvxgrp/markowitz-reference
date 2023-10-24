import numpy as np


def synthetic_returns(prices, var_r=0.0005, var_eps=0.02):
    """
    param prices: a DataFrame of prices
    param var_r: the Gaussian variance of the returns
    param var_eps: the Gaussian variance of the noise term

    returns: a DataFrame of "synthetic return predictions" computed as
    alpha*(returns+noise), where alpha=var_r / (var_r + var_eps); this is the
    coefficient that minimize the variance of the prediction error under the
    above model.

    var_r = 0.0005 and var_eps = 0.02 correspond to an information ratio
    sqrt(alpha) of about 0.15.
    """
    returns = prices.pct_change()

    alpha = var_r / (var_r + var_eps)
    sigma_eps = np.sqrt(var_eps)
    return alpha * (returns + np.random.normal(size=returns.shape) * sigma_eps)
