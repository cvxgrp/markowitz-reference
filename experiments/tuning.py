import numpy as np
import pandas as pd
from markowitz import Data, Parameters, markowitz
from dataclasses import dataclass

from backtest import run_markowitz, load_data


@dataclass
class HyperParameters:
    gamma_hold: float
    gamma_trade: float
    gamma_turn: float
    gamma_leverage: float
    gamma_risk: float


@dataclass
class Targets:
    T_target: float
    L_target: float
    risk_target: float


@dataclass
class Limits:
    T_max: float
    L_max: float
    risk_max: float


def get_data_and_parameters(
    inputs: callable, hyperparameters: dataclass, targets: dataclass, limits: dataclass
):
    """
    param inputs: OptimizationInputs object
    param hyperparameters: HyperParameters object (gamma_hold, gamma_trade,
    gamma_turn, gamma_leverage, gamma_risk)
    param targets: Targets object (T_target, L_target, risk_target)
    param limits: Limits object (T_max, L_max, risk_max)
    """

    # TODO: correct??? We do not know the spread and volume of the last day,
    # right? It's a bit weird, becaus we assume knowledge of the price...
    latest_prices = inputs.prices.iloc[-1]
    portfolio_value = inputs.cash + inputs.quantities @ latest_prices

    spread_prediction = inputs.spread.iloc[-6:-1].mean().values

    # print(111, (pd.Series(volume_prediction) == 0).sum())
    # print(1, volatilities)
    # print(2, volume_prediction)
    # print(3, kappa_impact)
    # print(43243, portfolio_value)

    # print(kappa_impact)

    w_lower = -0.05
    w_upper = 0.1
    c_lower = -0.05
    c_upper = 1
    z_lower = -0.1
    z_upper = 0.1

    rho_covariance = 0.02
    rho_mean = inputs.mean.abs().quantile(0.2)

    n_assets = inputs.n_assets

    # Hyperparameters
    t = latest_prices.name
    gamma_hold = hyperparameters.gamma_hold
    gamma_trade = hyperparameters.gamma_trade

    if type(hyperparameters.gamma_turn) == pd.Series:
        gamma_turn = hyperparameters.gamma_turn.loc[t]
    else:
        gamma_turn = hyperparameters.gamma_turn
    if type(hyperparameters.gamma_risk) == pd.Series:
        gamma_risk = hyperparameters.gamma_risk.loc[t]
    else:
        gamma_risk = hyperparameters.gamma_risk
    if type(hyperparameters.gamma_leverage) == pd.Series:
        gamma_leverage = hyperparameters.gamma_leverage.loc[t]
    else:
        gamma_leverage = hyperparameters.gamma_leverage

    data = Data(
        w_prev=(inputs.quantities * latest_prices / portfolio_value),
        c_prev=(inputs.cash / portfolio_value),
        idio_mean=np.zeros(n_assets),
        factor_mean=inputs.mean.values,
        risk_free=inputs.risk_free,
        factor_covariance_chol=np.linalg.cholesky(inputs.covariance.values),
        idio_volas=np.zeros(n_assets),
        F=np.eye(n_assets),
        kappa_short=np.ones(n_assets) * 3 * (0.01) ** 2,  # 7.5% yearly
        kappa_borrow=inputs.risk_free,
        kappa_spread=np.ones(n_assets) * spread_prediction / 2,
        kappa_impact=np.zeros(n_assets),
    )

    param = Parameters(
        w_lower=w_lower,
        w_upper=w_upper,
        c_lower=c_lower,
        c_upper=c_upper,
        z_lower=z_lower * np.ones(data.n_assets),
        z_upper=z_upper * np.ones(data.n_assets),
        T_target=targets.T_target,
        T_max=limits.T_max,
        L_target=targets.L_target,
        L_max=limits.L_max,
        rho_mean=np.ones(n_assets) * rho_mean,
        rho_covariance=rho_covariance,
        gamma_hold=gamma_hold,
        gamma_trade=gamma_trade,
        gamma_turn=gamma_turn,
        gamma_risk=gamma_risk,
        risk_max=limits.risk_max,
        risk_target=targets.risk_target,
        gamma_leverage=gamma_leverage,
    )

    return data, param


def full_markowitz(inputs, hyperparamters, targets, limits):
    """
    param inputs: OptimizationInputs object
    param hyperparameters: HyperParameters object (gamma_hold, gamma_trade,
    gamma_turn, gamma_leverage, gamma_risk)
    param targets: Targets object (T_target, L_target, risk_target)
    param limits: Limits object (T_max, L_max, risk_max)

    returns: w, c, problem, problem_solved
    """

    data, param = get_data_and_parameters(inputs, hyperparamters, targets, limits)

    w, c, problem, problem_solved = markowitz(data, param)
    return w, c, problem, problem_solved


def main():
    gamma_risk = 0.05
    gamma_turn = 0.002
    gamma_leverage = 0.0005

    prices, _, _, _ = load_data()

    gamma_turns = pd.Series(np.ones(len(prices)), index=prices.index) * gamma_turn
    gamma_leverages = (
        pd.Series(np.ones(len(prices)), index=prices.index) * gamma_leverage
    )
    gamma_risks = pd.Series(np.ones(len(prices)), index=prices.index) * gamma_risk

    hyperparameters = HyperParameters(1, 1, gamma_turns, gamma_leverages, gamma_risks)

    targets = Targets(
        T_target=50 / 252,
        L_target=1.6,
        risk_target=0.1 / np.sqrt(252),
    )
    limits = Limits(
        T_max=1e3,
        L_max=1e3,
        risk_max=1e3,
    )

    results, _ = run_markowitz(
        full_markowitz,
        targets=targets,
        limits=limits,
        hyperparameters=hyperparameters,
        verbose=True,
    )

    metrics = pd.Series(
        index=[
            "Mean return",
            "Volatility",
            "Sharpe",
            "Turnover",
            "Leverage",
            "Drawdown",
        ]
    )

    metrics["Mean return"] = results.mean_return
    metrics["Volatility"] = results.volatility
    metrics["Sharpe"] = results.sharpe
    metrics["Turnover"] = results.turnover
    metrics["Leverage"] = results.max_leverage
    metrics["Drawdown"] = results.max_drawdown

    print(metrics)
    # save file
    metrics.to_csv("full_markowitz_metrics.csv")


if __name__ == "__main__":
    main()
