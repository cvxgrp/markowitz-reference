import os
import numpy as np
import pandas as pd
import cvxpy as cp
from backtest import BacktestResult, OptimizationInput, run_backtest
from markowitz import Parameters
import matplotlib.pyplot as plt


def unconstrained_markowitz(inputs: OptimizationInput) -> np.ndarray:
    """Compute the unconstrained Markowitz portfolio weights."""
    n_assets = inputs.prices.shape[1]
    # mu, Sigma = ewma_mean_covariance(inputs.prices)

    mu, Sigma = inputs.mean.values, inputs.covariance.values

    w = cp.Variable(n_assets)
    c = cp.Variable()
    objective = mu @ w

    chol = np.linalg.cholesky(Sigma)
    constraints = [
        cp.sum(w) + c == 1,
        # cp.quad_form(w, Sigma, assume_PSD=True) <= inputs.risk_target ** 2
        cp.norm2(chol @ w) <= inputs.risk_target,  # faster in my experience
    ]
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(get_solver())
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    return w.value, c.value


def long_only_markowitz(inputs: OptimizationInput) -> np.ndarray:
    """Compute the long-only Markowitz portfolio weights."""
    n_assets = inputs.prices.shape[1]
    # mu, Sigma = ewma_mean_covariance(inputs.prices)

    mu, Sigma = inputs.mean.values, inputs.covariance.values

    w = cp.Variable(n_assets, nonneg=True)
    c = cp.Variable(nonneg=True)
    objective = mu @ w
    constraints = [
        cp.sum(w) + c == 1,
        cp.quad_form(w, Sigma, assume_PSD=True) <= inputs.risk_target**2,
    ]
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(get_solver())
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    return w.value, c.value


def equal_weights(inputs: OptimizationInput) -> np.ndarray:
    """Compute the equal weights portfolio."""
    n_assets = inputs.prices.shape[1]
    w = np.ones(n_assets) / (n_assets + 1)
    c = 1 / (n_assets + 1)
    return w, c


# def prepare_data(
#
# prices: pd.DataFrame, spread:
#  pd.DataFrame, volume: pd.DataFrame, quantities: np.ndarray, cash: float
#     ) -> Data:
#     n_assets = prices.shape[1]
#     latest_prices = prices.iloc[-1]
#     portfolio_value = cash + quantities @ latest_prices

#     mu, Sigma = ewma_mean_covariance(prices)

#     return Data(
#         w_prev = quantities * latest_prices / portfolio_value,
#         c_prev = cash / portfolio_value,
#         idio_mean = mu,
#         factor_mean = np.zeros(n_assets),
#         risk_free = 0,
#         factor_covariance_chol = np.linalg.cholesky(Sigma),
#         idio_volas = np.sqrt(np.diag(Sigma)),
#         F = np.eye(n_assets),
#         kappa_short = np.zeros(n_assets),
#         kappa_borrow = 0.0,
#         kappa_spread = np.zeros(n_assets),
#         kappa_impact = np.zeros(n_assets),
#     )


def get_parameters(data, risk_target):
    return Parameters(
        w_lower=np.zeros(data.n_assets),
        w_upper=np.ones(data.n_assets),
        c_lower=0.0,
        c_upper=1.0,
        z_lower=-np.ones(data.n_assets),
        z_upper=np.ones(data.n_assets),
        T_max=0.1,
        L_max=1.5,
        rho_mean=np.zeros(data.n_assets),
        rho_covariance=0.0,
        gamma_hold=0.0,
        gamma_trade=0.0,
        gamma_turn=0.0,
        gamma_risk=0.0,
        risk_target=risk_target,
    )


def main(from_checkpoint: bool = False):
    if from_checkpoint:
        unconstrained_results = []
        for f in [
            f for f in os.listdir("checkpoints") if f.startswith("unconstrained")
        ]:
            unconstrained_results.append(BacktestResult.load(f"checkpoints/{f}"))
        equal_weights_results = BacktestResult.load("checkpoints/equal_weights.pickle")
    else:
        equal_weights_results = run_backtest(equal_weights, 0.0, verbose=True)
        equal_weights_results.save("checkpoints/equal_weights.pickle")

        adjustment_factor = np.sqrt(equal_weights_results.periods_per_year)
        sigma_targets = np.array([0.10]) / adjustment_factor
        unconstrained_results = []
        for sigma_target in sigma_targets:
            result = run_backtest(unconstrained_markowitz, sigma_target, verbose=True)
            result.save(
                f"checkpoints/unconstrained_{result.risk_target * adjustment_factor:.2f}.pickle"
            )
            unconstrained_results.append(result)

        long_only_results = []
        for sigma_target in sigma_targets:
            result = run_backtest(long_only_markowitz, sigma_target, verbose=True)
            result.save(
                f"checkpoints/long_only_{result.risk_target * adjustment_factor:.2f}.pickle"
            )
            long_only_results.append(result)

    generate_table(equal_weights_results, unconstrained_results, long_only_results)
    plot_results(equal_weights_results, unconstrained_results, long_only_results)


def generate_table(
    equal_weights_results: BacktestResult,
    unconstrained_results: list[BacktestResult],
    long_only_results: list[BacktestResult],
) -> None:
    # Table 1
    df = pd.DataFrame(
        index=["Equal weights"]
        + [
            f"$\\sigma^\\text{{tar}} = {result.risk_target:.2f}$"
            for result in unconstrained_results
        ],
        columns=["Mean return", "Volatility", "Sharpe", "Turnover", "Max leverage"],
    )
    df["Mean return"] = [equal_weights_results.mean_return] + [
        result.mean_return for result in unconstrained_results
    ]
    df["Volatility"] = [equal_weights_results.volatility] + [
        result.volatility for result in unconstrained_results
    ]
    df["Sharpe"] = [equal_weights_results.sharpe] + [
        result.sharpe for result in unconstrained_results
    ]
    df["Turnover"] = [equal_weights_results.turnover] + [
        result.turnover for result in unconstrained_results
    ]
    df["Max leverage"] = [
        equal_weights_results.asset_weights.abs().sum(axis=1).max()
    ] + [
        result.asset_weights.abs().sum(axis=1).max() for result in unconstrained_results
    ]
    print(df.to_latex(float_format="%.2f"))

    # Table 2
    df = pd.DataFrame(
        index=[
            f"$\\sigma^\\text{{tar}} = {result.risk_target:.2f}$"
            for result in long_only_results
        ],
        columns=["Mean return", "Volatility", "Sharpe", "Turnover", "Max leverage"],
    )
    df["Mean return"] = [result.mean_return for result in long_only_results]
    df["Volatility"] = [result.volatility for result in long_only_results]
    df["Sharpe"] = [result.sharpe for result in long_only_results]
    df["Turnover"] = [result.turnover for result in long_only_results]
    df["Max leverage"] = [
        result.asset_weights.abs().sum(axis=1).max() for result in long_only_results
    ]
    print(df.to_latex(float_format="%.2f"))


def plot_results(
    equal_weights_results: BacktestResult,
    unconstrained_results: list[BacktestResult],
    long_only_results: list[BacktestResult],
) -> None:
    # E-V plot
    plt.figure()

    # Single star for equal weights
    plt.scatter(
        equal_weights_results.volatility,
        equal_weights_results.mean_return,
        marker="*",
        s=200,
        c="r",
        label="Equal weights",
    )

    # Circle for unconstrained Markowitz frontier as line with circles
    unconstrained_volatility = [result.volatility for result in unconstrained_results]
    unconstrained_mean_return = [result.mean_return for result in unconstrained_results]
    plt.plot(
        unconstrained_volatility,
        unconstrained_mean_return,
        "o-",
        label="Unconstrained Markowitz",
    )

    plt.xlabel("Volatility")
    plt.ylabel("Mean return")
    plt.legend()
    plt.show()


def get_solver():
    return cp.MOSEK if cp.MOSEK in cp.installed_solvers() else cp.CLARABEL


if __name__ == "__main__":
    main()
