import numpy as np
import pandas as pd
import cvxpy as cp
from backtest import BacktestResult, OptimizationInput, run_backtest
from markowitz import Data, Parameters, markowitz


def unconstrained_markowitz(
    inputs: OptimizationInput, long_only: bool = False
) -> np.ndarray:
    """Compute the unconstrained (or long-only) Markowitz portfolio weights."""

    mu, Sigma = inputs.mean.values, inputs.covariance.values

    if long_only:
        w = cp.Variable(inputs.n_assets, nonneg=True)
        c = cp.Variable(nonneg=True)
    else:
        w = cp.Variable(inputs.n_assets)
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
    return w.value, c.value, problem


def long_only_markowitz(inputs: OptimizationInput) -> np.ndarray:
    """Compute the long-only Markowitz portfolio weights."""
    return unconstrained_markowitz(inputs, long_only=True)


def equal_weights(inputs: OptimizationInput) -> np.ndarray:
    """Compute the equal weights portfolio."""
    n_assets = inputs.prices.shape[1]
    w = np.ones(n_assets) / (n_assets + 1)
    c = 1 / (n_assets + 1)
    return w, c, None


def weight_limits_markowitz(inputs: OptimizationInput) -> np.ndarray:
    lb = np.ones(inputs.n_assets) * (-0.05)
    ub = np.ones(inputs.n_assets) * 0.1

    data, param = get_unconstrained_data_and_parameters(inputs)

    param.w_lower = lb
    param.w_upper = ub
    param.c_lower = -0.05
    param.c_upper = 1.0
    param.risk_target = inputs.risk_target
    param.gamma_risk = 5.0
    return markowitz(data, param)


def leverage_limit_markowitz(inputs: OptimizationInput) -> np.ndarray:
    data, param = get_unconstrained_data_and_parameters(inputs)

    param.L_max = 1.6
    return markowitz(data, param)


def turnover_limit_markowitz(inputs: OptimizationInput) -> np.ndarray:
    data, param = get_unconstrained_data_and_parameters(inputs)

    param.T_max = 50 / 252  # Maximum TO per year
    return markowitz(data, param)


def robust_markowitz(inputs: OptimizationInput) -> np.ndarray:
    data, param = get_unconstrained_data_and_parameters(inputs)
    param.rho_mean = np.percentile(np.abs(inputs.mean.values), 20, axis=0) * np.ones(
        inputs.n_assets
    )
    param.rho_covariance = 0.02
    return markowitz(data, param)


def cost_aware_markowitz(inputs: OptimizationInput) -> np.ndarray:
    data, param = get_unconstrained_data_and_parameters(inputs)

    data.kappa_spread = (inputs.spread.iloc[-1] * 0.5).values
    data.risk_free = inputs.risk_free
    param.gamma_trade = 10.0
    return markowitz(data, param)


def get_unconstrained_data_and_parameters(
    inputs: OptimizationInput,
) -> tuple[Data, Parameters]:
    n_assets = inputs.n_assets
    latest_prices = inputs.prices.iloc[-1]
    portfolio_value = inputs.cash + inputs.quantities @ latest_prices

    # The risk constraint is soft.
    # For each percentage point of risk, we need to compensate with
    # 5 percentage points of return.
    gamma_risk = 5.0

    data = Data(
        w_prev=(inputs.quantities * latest_prices / portfolio_value).values,
        c_prev=(inputs.cash / portfolio_value),
        idio_mean=np.zeros(n_assets),
        factor_mean=inputs.mean.values,
        risk_free=0,
        factor_covariance_chol=np.linalg.cholesky(inputs.covariance.values),
        idio_volas=np.zeros(n_assets),
        F=np.eye(n_assets),
        kappa_short=np.zeros(n_assets),
        kappa_borrow=0.0,
        kappa_spread=np.zeros(n_assets),
        kappa_impact=np.zeros(n_assets),
    )
    param = Parameters(
        w_lower=-np.ones(data.n_assets) * 1e3,
        w_upper=np.ones(data.n_assets) * 1e3,
        c_lower=-1e3,
        c_upper=1e3,
        z_lower=-np.ones(data.n_assets) * 1e3,
        z_upper=np.ones(data.n_assets) * 1e3,
        T_max=1e3,
        L_max=1e3,
        rho_mean=np.zeros(data.n_assets),
        rho_covariance=0.0,
        gamma_hold=0.0,
        gamma_trade=0.0,
        gamma_turn=0.0,
        gamma_risk=gamma_risk,
        risk_target=inputs.risk_target,
    )
    return data, param


def main(from_checkpoint: bool = False):
    annualized_target = 0.10

    if not from_checkpoint:
        run_all_strategies(annualized_target)

    equal_weights_results = BacktestResult.load("checkpoints/equal_weights.pickle")

    unconstrained_result = BacktestResult.load(
        f"checkpoints/unconstrained_{annualized_target}.pickle"
    )
    weight_limited_result = BacktestResult.load(
        f"checkpoints/weight_limited_{annualized_target}.pickle"
    )
    leverage_limit_result = BacktestResult.load(
        f"checkpoints/leverage_limit_{annualized_target}.pickle"
    )
    turnover_limit_result = BacktestResult.load(
        f"checkpoints/turnover_limit_{annualized_target}.pickle"
    )
    robust_result = BacktestResult.load(
        f"checkpoints/robust_{annualized_target}.pickle"
    )

    cost_aware_result = BacktestResult.load(
        f"checkpoints/cost_aware_{annualized_target}.pickle"
    )

    generate_table(
        equal_weights_results,
        unconstrained_result,
        weight_limited_result,
        leverage_limit_result,
        turnover_limit_result,
        robust_result,
        cost_aware_result,
    )

    # plot_timings(robust_result.timings)
    # plot_results(equal_weights_results, unconstrained_result, long_only_result)


def run_all_strategies(annualized_target: float) -> None:
    equal_weights_results = run_backtest(equal_weights, 0.0, verbose=True)
    equal_weights_results.save("checkpoints/equal_weights.pickle")

    adjustment_factor = np.sqrt(equal_weights_results.periods_per_year)
    sigma_target = annualized_target / adjustment_factor

    print("Running unconstrained Markowitz")
    unconstrained_result = run_backtest(
        unconstrained_markowitz, sigma_target, verbose=True
    )
    unconstrained_result.save(f"checkpoints/unconstrained_{annualized_target}.pickle")

    print("Running leverage limit Markowitz")
    leverage_limit_result = run_backtest(
        leverage_limit_markowitz, sigma_target, verbose=True
    )
    leverage_limit_result.save(f"checkpoints/leverage_limit_{annualized_target}.pickle")

    print("Running turnover limit Markowitz")
    turnover_limit_result = run_backtest(
        turnover_limit_markowitz, sigma_target, verbose=True
    )
    turnover_limit_result.save(f"checkpoints/turnover_limit_{annualized_target}.pickle")

    print("Running cost-aware Markowitz")
    cost_aware_result = run_backtest(cost_aware_markowitz, sigma_target, verbose=True)
    cost_aware_result.save(f"checkpoints/cost_aware_{annualized_target}.pickle")

    print("Running robust Markowitz")
    robust_result = run_backtest(robust_markowitz, sigma_target, verbose=True)
    robust_result.save(f"checkpoints/robust_{annualized_target}.pickle")

    print("Running weight-limited Markowitz")
    weight_limited_result = run_backtest(
        weight_limits_markowitz, sigma_target, verbose=True
    )
    weight_limited_result.save(f"checkpoints/weight_limited_{annualized_target}.pickle")


def generate_table(
    equal_weights_results: BacktestResult,
    unconstrained_results: BacktestResult,
    weight_limited_result: BacktestResult,
    leverage_limit_result: BacktestResult,
    turnover_limit_result: BacktestResult,
    robust_result: BacktestResult,
    cost_aware_result: BacktestResult,
) -> None:
    # Table 1
    df = pd.DataFrame(
        index=[
            "Equal weight",
            "Basic Markowitz",
            "Weight-limited",
            "Leverage-limited",
            "Turnover-limited",
            "Robust",
            "Cost-aware",
        ],
        columns=[
            "Mean return",
            "Volatility",
            "Sharpe",
            "Turnover",
            "Leverage",
            "Drawdown",
        ],
    )
    strategies = [
        equal_weights_results,
        unconstrained_results,
        weight_limited_result,
        leverage_limit_result,
        turnover_limit_result,
        robust_result,
        cost_aware_result,
    ]

    df["Mean return"] = [result.mean_return for result in strategies]
    df["Volatility"] = [result.volatility for result in strategies]
    df["Sharpe"] = [result.sharpe for result in strategies]
    df["Turnover"] = [result.turnover for result in strategies]
    df["Leverage"] = [result.max_leverage for result in strategies]
    df["Drawdown"] = [result.max_drawdown for result in strategies]

    formatters = {
        "Mean return": lambda x: rf"{100 * x:.1f}\%",
        "Volatility": lambda x: rf"{100 * x:.1f}\%",
        "Sharpe": lambda x: f"{x:.2f}",
        "Turnover": lambda x: f"{x:.1f}",
        "Leverage": lambda x: f"{x:.1f}",
        "Drawdown": lambda x: rf"{100 * x:.1f}\%",
    }

    print(
        df.to_latex(
            formatters=formatters,
        )
    )


def get_solver():
    return cp.MOSEK if cp.MOSEK in cp.installed_solvers() else cp.CLARABEL


if __name__ == "__main__":
    main()
