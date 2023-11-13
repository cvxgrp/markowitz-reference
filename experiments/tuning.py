import numpy as np
import cvxpy as cp
from markowitz import Data, Parameters, markowitz
from dataclasses import dataclass


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
    inputs: callable, hyperparameters: callable, targets: callable, limits: callable
):
    # L_max = 1.6
    # T_max = 10 / 252
    # risk_target = inputs.risk_target

    spread_prediction = inputs.spread.iloc[-5:].mean().values

    w_lower = -0.1
    w_upper = 0.15
    c_lower = -0.3
    c_upper = 0.4
    z_lower = -0.1
    z_upper = 0.1

    rho_covariance = 0.02
    rho_mean = 0.0

    n_assets = inputs.n_assets
    latest_prices = inputs.prices.iloc[-1]
    portfolio_value = inputs.cash + inputs.quantities @ latest_prices

    # Hyperparameters
    t = latest_prices.name
    gamma_hold = hyperparameters.gamma_hold
    gamma_trade = hyperparameters.gamma_trade
    gamma_turn = hyperparameters.gamma_turn.loc[t]
    gamma_risk = hyperparameters.gamma_risk.loc[t]
    gamma_leverage = hyperparameters.gamma_leverage.loc[t]

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


def initialize_markowitz(inputs, hyperparameters, targets, limits):
    n_assets = inputs.n_assets

    w_lower = -0.1
    w_upper = 0.15
    c_lower = 0
    c_upper = 0
    z_lower = -1e3
    z_upper = 1e3

    rho_covariance = 0.02
    rho_mean = 0.0

    data = Data(
        w_prev=(1 / n_assets) * np.ones(n_assets),
        c_prev=0,
        idio_mean=np.zeros(n_assets),
        factor_mean=np.zeros(n_assets),
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
        w_lower=w_lower,
        w_upper=w_upper,
        c_lower=c_lower,
        c_upper=c_upper,
        z_lower=z_lower * np.ones(data.n_assets),
        z_upper=z_upper * np.ones(data.n_assets),
        T_max=1e3,
        L_max=limits.L_max,
        rho_mean=np.ones(n_assets) * rho_mean,
        rho_covariance=rho_covariance,
        gamma_hold=0,
        gamma_trade=0,
        gamma_turn=0,
        gamma_risk=1,
        risk_target=0,
        gamma_leverage=0,
        risk_max=1e3,
        T_target=0,
        L_target=0,
    )

    w, c, problem, problem_solved = markowitz(data, param)
    return w, c, problem, problem_solved


def full_markowitz(inputs, hyperparamters, targets, limits, initialize=False):
    if initialize:
        return initialize_markowitz(inputs, hyperparamters, targets, limits)

    data, param = get_data_and_parameters(inputs, hyperparamters, targets, limits)
    try:
        w, c, problem, problem_solved = markowitz(data, param)
        return w, c, problem, problem_solved
    except cp.SolverError:
        # print("Failed to solve markowitz")
        return data.w_prev, data.c_prev, None, False
