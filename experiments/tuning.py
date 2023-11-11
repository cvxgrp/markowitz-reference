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


def get_data_and_parameters(inputs: callable, hyperparameters: callable):
    L_max = 1.6
    T_max = 10 / 252
    risk_target = inputs.risk_target

    w_lower = -0.1
    w_upper = 0.15
    c_lower = -0.3
    c_upper = 1
    z_lower = -0.1
    z_upper = 0.1

    rho_covariance = 0.02
    rho_mean = 0.0

    n_assets = inputs.n_assets
    latest_prices = inputs.prices.iloc[-1]
    portfolio_value = inputs.cash + inputs.quantities @ latest_prices

    data = Data(
        w_prev=(inputs.quantities * latest_prices / portfolio_value),
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
        w_lower=w_lower,
        w_upper=w_upper,
        c_lower=c_lower,
        c_upper=c_upper,
        z_lower=z_lower * np.ones(data.n_assets),
        z_upper=z_upper * np.ones(data.n_assets),
        T_max=T_max,
        L_max=L_max,
        rho_mean=np.ones(n_assets) * rho_mean,
        rho_covariance=rho_covariance,
        gamma_hold=hyperparameters.gamma_hold,
        gamma_trade=hyperparameters.gamma_trade,
        gamma_turn=hyperparameters.gamma_turn,
        gamma_risk=hyperparameters.gamma_risk,
        risk_target=risk_target,
        gamma_leverage=hyperparameters.gamma_leverage,
    )

    return data, param


def initialize_markowitz(inputs, hyperparameters):
    n_assets = inputs.n_assets
    latest_prices = inputs.prices.iloc[-1]
    portfolio_value = inputs.cash + inputs.quantities @ latest_prices

    w_lower = -0.1
    w_upper = 0.15
    c_lower = 0
    c_upper = 0
    z_lower = -1e3
    z_upper = 1e3

    rho_covariance = 0.02
    rho_mean = 0.0

    data = Data(
        w_prev=(inputs.quantities * latest_prices / portfolio_value),
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
        w_lower=w_lower,
        w_upper=w_upper,
        c_lower=c_lower,
        c_upper=c_upper,
        z_lower=z_lower * np.ones(data.n_assets),
        z_upper=z_upper * np.ones(data.n_assets),
        T_max=1e3,
        L_max=1.6,
        rho_mean=np.ones(n_assets) * rho_mean,
        rho_covariance=rho_covariance,
        gamma_hold=hyperparameters.gamma_hold,
        gamma_trade=hyperparameters.gamma_trade,
        gamma_turn=hyperparameters.gamma_turn,
        gamma_risk=hyperparameters.gamma_risk,
        risk_target=inputs.risk_target,
        gamma_leverage=hyperparameters.gamma_leverage,
    )

    w, c, problem, problem_solved = markowitz(data, param)
    return w, c, problem, problem_solved


def full_markowitz(inputs, hyperparamters, initialize=False, hard=True):
    if not initialize:
        data, param = get_data_and_parameters(inputs, hyperparamters)
    else:
        return initialize_markowitz(inputs, hyperparamters)

    try:
        w, c, problem, problem_solved = markowitz(data, param, hard=hard)
        return w, c, problem, problem_solved
    except cp.SolverError:
        # print("Failed to solve markowitz")
        return data.w_prev, data.c_prev, None, False
