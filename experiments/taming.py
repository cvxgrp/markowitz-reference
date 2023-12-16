import numpy as np
import pandas as pd
import cvxpy as cp

from experiments.backtest import BacktestResult, OptimizationInput, run_backtest
from experiments.markowitz import Data, Parameters, markowitz
from experiments.utils import get_solver


def basic_markowitz(inputs: OptimizationInput) -> tuple[np.ndarray, float, cp.Problem]:
    """Compute the basic Markowitz portfolio weights."""

    w = cp.Variable(inputs.n_assets)
    c = cp.Variable()
    objective = inputs.mean.values @ w

    constraints = [
        cp.sum(w) + c == 1,
        cp.norm2(inputs.chol.T @ w) <= inputs.risk_target,
    ]

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(get_solver())
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}, problem.status
    return w.value, c.value, problem


def equal_weights(inputs: OptimizationInput) -> tuple[np.ndarray, float, cp.Problem]:
    """Compute the equal weights portfolio."""
    n_assets = inputs.prices.shape[1]
    w = np.ones(n_assets) / (n_assets + 1)
    c = 1 / (n_assets + 1)
    return w, c, None


def weight_limits_markowitz(
    inputs: OptimizationInput,
) -> tuple[np.ndarray, float, cp.Problem]:
    lb = np.ones(inputs.n_assets) * (-0.05)
    ub = np.ones(inputs.n_assets) * 0.1

    data, param = get_basic_data_and_parameters(inputs)

    param.w_min = lb
    param.w_max = ub
    param.c_min = -0.05
    param.c_max = 1.0
    param.risk_target = inputs.risk_target
    return markowitz(data, param)


def leverage_limit_markowitz(
    inputs: OptimizationInput,
) -> tuple[np.ndarray, float, cp.Problem]:
    data, param = get_basic_data_and_parameters(inputs)

    param.L_tar = 1.6
    return markowitz(data, param)


def turnover_limit_markowitz(
    inputs: OptimizationInput,
) -> tuple[np.ndarray, float, cp.Problem]:
    data, param = get_basic_data_and_parameters(inputs)

    param.T_tar = 50 / 252  # Maximum TO per year
    return markowitz(data, param)


def robust_markowitz(inputs: OptimizationInput) -> tuple[np.ndarray, float, cp.Problem]:
    data, param = get_basic_data_and_parameters(inputs)
    param.rho_mean = np.percentile(np.abs(inputs.mean.values), 20, axis=0) * np.ones(
        inputs.n_assets
    )
    param.rho_covariance = 0.02
    return markowitz(data, param)


def get_basic_data_and_parameters(
    inputs: OptimizationInput,
) -> tuple[Data, Parameters]:
    n_assets = inputs.n_assets
    latest_prices = inputs.prices.iloc[-1]
    portfolio_value = inputs.cash + inputs.quantities @ latest_prices

    data = Data(
        w_prev=(inputs.quantities * latest_prices / portfolio_value).values,
        idio_mean=np.zeros(n_assets),
        factor_mean=inputs.mean.values,
        risk_free=0,
        factor_covariance_chol=inputs.chol,
        idio_volas=np.zeros(n_assets),
        F=np.eye(n_assets),
        kappa_short=np.zeros(n_assets),
        kappa_borrow=0.0,
        kappa_spread=np.zeros(n_assets),
        kappa_impact=np.zeros(n_assets),
    )
    param = Parameters(
        w_min=-np.ones(data.n_assets) * 1e3,
        w_max=np.ones(data.n_assets) * 1e3,
        c_min=-1e3,
        c_max=1e3,
        z_min=-np.ones(data.n_assets) * 1e3,
        z_max=np.ones(data.n_assets) * 1e3,
        T_tar=1e3,
        L_tar=1e3,
        rho_mean=np.zeros(data.n_assets),
        rho_covariance=0.0,
        gamma_hold=0.0,
        gamma_trade=0.0,
        gamma_turn=0.0,
        gamma_risk=0.0,
        risk_target=inputs.risk_target,
        risk_max=1e3,
        gamma_leverage=0.0,
        T_target=0.0,
        L_target=0.0,
    )

    return data, param


def main(from_checkpoint: bool = False) -> None:
    annualized_target = 0.10

    if not from_checkpoint:
        run_all_strategies(annualized_target)

    equal_weights_results = BacktestResult.load("checkpoints/equal_weights.pickle")

    basic_result = BacktestResult.load(f"checkpoints/basic_{annualized_target}.pickle")

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

    generate_table(
        equal_weights_results,
        basic_result,
        weight_limited_result,
        leverage_limit_result,
        turnover_limit_result,
        robust_result,
    )

    generate_per_year_tables(
        equal_weights_results,
        basic_result,
        weight_limited_result,
        leverage_limit_result,
        turnover_limit_result,
        robust_result,
    )


def run_all_strategies(annualized_target: float) -> None:
    equal_weights_results = run_backtest(equal_weights, 0.0, verbose=True)
    equal_weights_results.save("checkpoints/equal_weights.pickle")

    adjustment_factor = np.sqrt(equal_weights_results.periods_per_year)
    sigma_target = annualized_target / adjustment_factor

    print("Running basic Markowitz")
    basic_result = run_backtest(basic_markowitz, sigma_target, verbose=True)
    basic_result.save(f"checkpoints/basic_{annualized_target}.pickle")

    print("Running weight-limited Markowitz")
    weight_limited_result = run_backtest(
        weight_limits_markowitz, sigma_target, verbose=True
    )
    weight_limited_result.save(f"checkpoints/weight_limited_{annualized_target}.pickle")

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

    print("Running robust Markowitz")
    robust_result = run_backtest(robust_markowitz, sigma_target, verbose=True)
    robust_result.save(f"checkpoints/robust_{annualized_target}.pickle")


def generate_table(
    equal_weights_results: BacktestResult,
    basic_results: BacktestResult,
    weight_limited_result: BacktestResult,
    leverage_limit_result: BacktestResult,
    turnover_limit_result: BacktestResult,
    robust_result: BacktestResult,
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
        basic_results,
        weight_limited_result,
        leverage_limit_result,
        turnover_limit_result,
        robust_result,
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
        "Drawdown": lambda x: rf"{-100 * x:.1f}\%",
    }

    print(
        df.to_latex(
            formatters=formatters,
        )
    )


def generate_per_year_tables(
    equal_weights_results: BacktestResult,
    basic_results: BacktestResult,
    weight_limited_result: BacktestResult,
    leverage_limit_result: BacktestResult,
    turnover_limit_result: BacktestResult,
    robust_result: BacktestResult,
) -> None:
    # TODO: DRY

    years = sorted(equal_weights_results.history.year.unique())

    equal_subs = []
    basic_subs = []
    weight_limited_subs = []
    leverage_limit_subs = []
    turnover_limit_subs = []
    robust_subs = []

    for year in years:
        sub_index = equal_weights_results.history.year == year

        equal_subs.append(get_sub_result(equal_weights_results, sub_index))
        basic_subs.append(get_sub_result(basic_results, sub_index))
        weight_limited_subs.append(get_sub_result(weight_limited_result, sub_index))
        leverage_limit_subs.append(get_sub_result(leverage_limit_result, sub_index))
        turnover_limit_subs.append(get_sub_result(turnover_limit_result, sub_index))
        robust_subs.append(get_sub_result(robust_result, sub_index))

    mean_df = pd.concat(
        [
            pd.Series(
                [result.mean_return for result in equal_subs], index=years, name="Equal"
            ),
            pd.Series(
                [result.mean_return for result in basic_subs], index=years, name="Basic"
            ),
            pd.Series(
                [result.mean_return for result in weight_limited_subs],
                index=years,
                name="Weight-limited",
            ),
            pd.Series(
                [result.mean_return for result in leverage_limit_subs],
                index=years,
                name="Leverage-limited",
            ),
            pd.Series(
                [result.mean_return for result in turnover_limit_subs],
                index=years,
                name="Turnover-limited",
            ),
            pd.Series(
                [result.mean_return for result in robust_subs],
                index=years,
                name="Robust",
            ),
        ],
        axis=1,
    )

    print(mean_df.to_latex(float_format=lambda x: rf"{100 * x:.1f}\%"))

    volatility_df = pd.concat(
        [
            pd.Series(
                [result.volatility for result in equal_subs], index=years, name="Equal"
            ),
            pd.Series(
                [result.volatility for result in basic_subs], index=years, name="Basic"
            ),
            pd.Series(
                [result.volatility for result in weight_limited_subs],
                index=years,
                name="Weight-limited",
            ),
            pd.Series(
                [result.volatility for result in leverage_limit_subs],
                index=years,
                name="Leverage-limited",
            ),
            pd.Series(
                [result.volatility for result in turnover_limit_subs],
                index=years,
                name="Turnover-limited",
            ),
            pd.Series(
                [result.volatility for result in robust_subs],
                index=years,
                name="Robust",
            ),
        ],
        axis=1,
    )

    print(volatility_df.to_latex(float_format=lambda x: rf"{100 * x:.1f}\%"))

    sharpe_df = pd.concat(
        [
            pd.Series(
                [result.sharpe for result in equal_subs], index=years, name="Equal"
            ),
            pd.Series(
                [result.sharpe for result in basic_subs], index=years, name="Basic"
            ),
            pd.Series(
                [result.sharpe for result in weight_limited_subs],
                index=years,
                name="Weight-limited",
            ),
            pd.Series(
                [result.sharpe for result in leverage_limit_subs],
                index=years,
                name="Leverage-limited",
            ),
            pd.Series(
                [result.sharpe for result in turnover_limit_subs],
                index=years,
                name="Turnover-limited",
            ),
            pd.Series(
                [result.sharpe for result in robust_subs], index=years, name="Robust"
            ),
        ],
        axis=1,
    )

    print(sharpe_df.to_latex(float_format=lambda x: f"{x:.2f}"))

    turnover_df = pd.concat(
        [
            pd.Series(
                [result.turnover for result in equal_subs], index=years, name="Equal"
            ),
            pd.Series(
                [result.turnover for result in basic_subs], index=years, name="Basic"
            ),
            pd.Series(
                [result.turnover for result in weight_limited_subs],
                index=years,
                name="Weight-limited",
            ),
            pd.Series(
                [result.turnover for result in leverage_limit_subs],
                index=years,
                name="Leverage-limited",
            ),
            pd.Series(
                [result.turnover for result in turnover_limit_subs],
                index=years,
                name="Turnover-limited",
            ),
            pd.Series(
                [result.turnover for result in robust_subs], index=years, name="Robust"
            ),
        ],
        axis=1,
    )

    print(turnover_df.to_latex(float_format=lambda x: f"{x:.1f}"))

    leverage_df = pd.concat(
        [
            pd.Series(
                [result.max_leverage for result in equal_subs],
                index=years,
                name="Equal",
            ),
            pd.Series(
                [result.max_leverage for result in basic_subs],
                index=years,
                name="Basic",
            ),
            pd.Series(
                [result.max_leverage for result in weight_limited_subs],
                index=years,
                name="Weight-limited",
            ),
            pd.Series(
                [result.max_leverage for result in leverage_limit_subs],
                index=years,
                name="Leverage-limited",
            ),
            pd.Series(
                [result.max_leverage for result in turnover_limit_subs],
                index=years,
                name="Turnover-limited",
            ),
            pd.Series(
                [result.max_leverage for result in robust_subs],
                index=years,
                name="Robust",
            ),
        ],
        axis=1,
    )

    print(leverage_df.to_latex(float_format=lambda x: f"{x:.1f}"))

    drawdown_df = pd.concat(
        [
            pd.Series(
                [result.max_drawdown for result in equal_subs],
                index=years,
                name="Equal",
            ),
            pd.Series(
                [result.max_drawdown for result in basic_subs],
                index=years,
                name="Basic",
            ),
            pd.Series(
                [result.max_drawdown for result in weight_limited_subs],
                index=years,
                name="Weight-limited",
            ),
            pd.Series(
                [result.max_drawdown for result in leverage_limit_subs],
                index=years,
                name="Leverage-limited",
            ),
            pd.Series(
                [result.max_drawdown for result in turnover_limit_subs],
                index=years,
                name="Turnover-limited",
            ),
            pd.Series(
                [result.max_drawdown for result in robust_subs],
                index=years,
                name="Robust",
            ),
        ],
        axis=1,
    )

    print(drawdown_df.to_latex(float_format=lambda x: rf"{-100 * x:.1f}\%"))


def get_sub_result(result: BacktestResult, sub_index: pd.Series) -> BacktestResult:
    return BacktestResult(
        cash=result.cash.loc[sub_index],
        quantities=result.quantities.loc[sub_index],
        risk_target=result.risk_target,
        timings=None,
    )


if __name__ == "__main__":
    main()
