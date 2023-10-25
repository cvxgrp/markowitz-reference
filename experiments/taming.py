import os
import numpy as np
import pandas as pd
import cvxpy as cp
from backtest import BacktestResult, OptimizationInput, run_backtest
from markowitz import Data, Parameters
import matplotlib.pyplot as plt


def unconstrained_markowitz(
    inputs: OptimizationInput, long_only: bool = False
) -> np.ndarray:
    """Compute the unconstrained (or long-only) Markowitz portfolio weights."""
    n_assets = inputs.prices.shape[1]

    mu, Sigma = inputs.mean.values, inputs.covariance.values

    if long_only:
        w = cp.Variable(n_assets, nonneg=True)
        c = cp.Variable(nonneg=True)
    else:
        w = cp.Variable(n_assets)
        c = cp.Variable()
    objective = mu @ w

    chol = np.linalg.cholesky(Sigma)
    constraints = [
        cp.sum(w) + c == 1,
        cp.norm2(chol.T @ w) <= inputs.risk_target,
    ]

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(get_solver())
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}, problem.status
    return w.value, c.value


def long_only_markowitz(inputs: OptimizationInput) -> np.ndarray:
    """Compute the long-only Markowitz portfolio weights."""
    return unconstrained_markowitz(inputs, long_only=True)


def equal_weights(inputs: OptimizationInput) -> np.ndarray:
    """Compute the equal weights portfolio."""
    n_assets = inputs.prices.shape[1]
    w = np.ones(n_assets) / (n_assets + 1)
    c = 1 / (n_assets + 1)
    return w, c


def ewma_mean_covariance(
    prices: pd.DataFrame, lamb: float = 0.94
) -> tuple[np.ndarray, np.ndarray]:
    returns = prices.pct_change().dropna()
    n_assets = returns.shape[1]
    alpha = 1 - lamb
    mu = returns.ewm(alpha=alpha).mean().iloc[-1].values
    Sigma = returns.ewm(alpha=alpha).cov().iloc[-n_assets:].values
    return mu, Sigma


def prepare_data(
    prices: pd.DataFrame,
    spread: pd.DataFrame,
    volume: pd.DataFrame,
    quantities: np.ndarray,
    cash: float,
) -> Data:
    n_assets = prices.shape[1]
    latest_prices = prices.iloc[-1]
    portfolio_value = cash + quantities @ latest_prices

    mu, Sigma = ewma_mean_covariance(prices)

    return Data(
        w_prev=quantities * latest_prices / portfolio_value,
        c_prev=cash / portfolio_value,
        idio_mean=mu,
        factor_mean=np.zeros(n_assets),
        risk_free=0,
        factor_covariance_chol=np.linalg.cholesky(Sigma),
        idio_volas=np.sqrt(np.diag(Sigma)),
        F=np.eye(n_assets),
        kappa_short=np.zeros(n_assets),
        kappa_borrow=0.0,
        kappa_spread=np.zeros(n_assets),
        kappa_impact=np.zeros(n_assets),
    )


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
        unconstrained_files = [
            f for f in os.listdir("checkpoints") if f.startswith("unconstrained")
        ]
        assert len(unconstrained_files) == 1
        unconstrained_result = BacktestResult.load(
            f"checkpoints/{unconstrained_files[0]}"
        )

        long_only_files = [
            f for f in os.listdir("checkpoints") if f.startswith("long_only")
        ]
        assert len(long_only_files) == 1
        long_only_result = BacktestResult.load(f"checkpoints/{long_only_files[0]}")

        equal_weights_results = BacktestResult.load("checkpoints/equal_weights.pickle")
    else:
        equal_weights_results = run_backtest(equal_weights, 0.0, verbose=True)
        equal_weights_results.save("checkpoints/equal_weights.pickle")

        adjustment_factor = np.sqrt(equal_weights_results.periods_per_year)
        annualized_target = 0.13
        sigma_target = annualized_target / adjustment_factor

        unconstrained_result = run_backtest(
            unconstrained_markowitz, sigma_target, verbose=True
        )
        unconstrained_result.save(
            f"checkpoints/unconstrained_{annualized_target}.pickle"
        )

        long_only_result = run_backtest(long_only_markowitz, sigma_target, verbose=True)
        long_only_result.save(f"checkpoints/long_only_{annualized_target}.pickle")

    generate_table(equal_weights_results, unconstrained_result, long_only_result)
    # plot_results(equal_weights_results, unconstrained_result, long_only_result)


def generate_table(
    equal_weights_results: BacktestResult,
    unconstrained_results: BacktestResult,
    long_only_results: BacktestResult,
) -> None:
    # Table 1
    df = pd.DataFrame(
        index=["Equal weights", "Unconstrained Markowitz", "Long-only Markowitz"],
        columns=["Mean return", "Volatility", "Sharpe", "Turnover", "Max leverage"],
    )
    strategies = [equal_weights_results, unconstrained_results, long_only_results]

    df["Mean return"] = [result.mean_return for result in strategies]
    df["Volatility"] = [result.volatility for result in strategies]
    df["Sharpe"] = [result.sharpe for result in strategies]
    df["Turnover"] = [result.turnover for result in strategies]
    df["Max leverage"] = [result.max_leverage for result in strategies]
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
