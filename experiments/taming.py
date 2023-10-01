import dataclasses
import multiprocessing
import numpy as np
import pandas as pd
import cvxpy as cp
from backtest import BacktestResult, OptimizationInput, run_backtest, load_data
from markowitz import Data, Parameters
import matplotlib.pyplot as plt


def unconstrained_markowitz(inputs: OptimizationInput) -> np.ndarray:
    """Compute the unconstrained Markowitz portfolio weights."""
    n_assets = inputs.prices.shape[1]
    mu, Sigma = ewma_mean_covariance(inputs.prices)

    w = cp.Variable(n_assets)    
    c = cp.Variable()
    objective = mu @ w
    constraints = [
        cp.sum(w) + c == 1, 
        cp.quad_form(w, Sigma, assume_PSD=True) <= inputs.risk_target ** 2
    ]
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(get_solver())
    assert problem.status == cp.OPTIMAL
    return w.value, c.value

def equal_weights(inputs: OptimizationInput) -> np.ndarray:
    """Compute the equal weights portfolio."""
    n_assets = inputs.prices.shape[1]
    w = np.ones(n_assets) / (n_assets + 1)
    c = 1 / (n_assets + 1)
    return w, c

def ewma_mean_covariance(prices: pd.DataFrame, lamb: float = 0.94) -> tuple[np.ndarray, np.ndarray]:
    returns = prices.pct_change().dropna()
    n_assets = returns.shape[1]
    alpha = 1 - lamb
    mu = returns.ewm(alpha=alpha).mean().iloc[-1].values
    Sigma = returns.ewm(alpha=alpha).cov().iloc[-n_assets:].values
    return mu, Sigma


def prepare_data(
        prices: pd.DataFrame, spread: pd.DataFrame, volume: pd.DataFrame, quantities: np.ndarray, cash: float
    ) -> Data:
    n_assets = prices.shape[1]
    latest_prices = prices.iloc[-1]
    portfolio_value = cash + quantities @ latest_prices

    mu, Sigma = ewma_mean_covariance(prices)

    return Data(
        w_prev = quantities * latest_prices / portfolio_value,
        c_prev = cash / portfolio_value,
        idio_mean = mu,
        factor_mean = np.zeros(n_assets),
        risk_free = 0,
        factor_covariance_chol = np.linalg.cholesky(Sigma),
        idio_volas = np.sqrt(np.diag(Sigma)),
        F = np.eye(n_assets),
        kappa_short = np.zeros(n_assets),
        kappa_borrow = 0.0,
        kappa_spread = np.zeros(n_assets),
        kappa_impact = np.zeros(n_assets),
    )


def get_parameters(data, risk_target):
    return Parameters(
        w_lower = np.zeros(data.n_assets),
        w_upper = np.ones(data.n_assets),
        c_lower = 0.0,
        c_upper = 1.0,
        z_lower = -np.ones(data.n_assets),
        z_upper = np.ones(data.n_assets),
        T_max = 0.1,
        L_max = 1.5,
        rho_mean = np.zeros(data.n_assets),
        rho_covariance = 0.0,
        gamma_hold = 0.0,
        gamma_trade = 0.0,
        gamma_turn = 0.0,
        gamma_risk = 0.0,
        risk_target = risk_target,
    )


def main(parallel: bool = True):
    sigma_targets = np.linspace(0.01, 0.21, 11)

    # Parallel computation of sigma targets
    if parallel:
        args = [(unconstrained_markowitz, sigma_target) for sigma_target in sigma_targets]
        with multiprocessing.Pool() as pool:
            unconstrained_results = pool.starmap(run_backtest, args)
    else:
        unconstrained_results = []
        for sigma_target in sigma_targets:
            unconstrained_results.append(run_backtest(unconstrained_markowitz, sigma_target, verbose=True))

    for result in unconstrained_results:
        result.save(f"checkpoints/unconstrained_{result.risk_target:.2f}.pickle")

    equal_weights_results = run_backtest(equal_weights, 0.0, verbose=True) 

    plot_results(equal_weights_results, unconstrained_results)

def plot_results(
        equal_weights_results: BacktestResult, 
        unconstrained_results: list[BacktestResult], 
        ) -> None:
    
    # E-V plot
    plt.figure()
    
    # Single star for equal weights
    plt.scatter(equal_weights_results.volatility, equal_weights_results.mean_return, marker="*", s=200, c="r", label="Equal weights")

    # Circle for unconstrained Markowitz frontier
    unconstrained_volatility = [result.volatility for result in unconstrained_results]
    unconstrained_mean_return = [result.mean_return for result in unconstrained_results]
    plt.scatter(unconstrained_volatility, unconstrained_mean_return, marker="o", s=100, c="b", label="Unconstrained Markowitz")
    
    plt.xlabel("Volatility")
    plt.ylabel("Mean return")
    plt.legend()
    plt.show()


def get_solver():
    return cp.MOSEK if cp.MOSEK in cp.installed_solvers() else cp.CLARABEL

if __name__ == "__main__":
    main()
