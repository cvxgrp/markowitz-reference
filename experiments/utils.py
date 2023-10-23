import numpy as np


def synthetic_returns(prices, sigma_r=0.02236, sigma_eps=0.14142):
    returns = prices.pct_change()

    alpha = sigma_r**2 / (sigma_r**2 + sigma_eps**2)
    return alpha * (returns + np.random.normal(size=returns.shape) * sigma_eps)
