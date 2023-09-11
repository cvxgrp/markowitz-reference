import pandas as pd
import cvxpy as cp
from dataclasses import dataclass

@dataclass(frozen=True)
class RiskModel:
    def __init__(self, factor_covariance, idio_vola, factor_exposures):
        """
        param factor_covariance: dictionary of (n_factors, n_factors) covariance
            matrices, where the keys are the dates
        param idio_vola: dataframe of idiosyncratic volatilities,
            where columns are assets and rows are the dates
        param factor_exposures: dictionary of factor exposures, where the keys
            are the dates
        """
        # make sure dates align
        assert set(factor_covariance.keys()) == set(idio_vola.index), \
            "factor_covariance and idio_vola must have the same dates"
        assert set(factor_covariance.keys()) == set(factor_exposures.keys()), \
            "factor_covariance and factor_exposures must have the same dates"

        self.factor_covariance = factor_covariance
        self.idio_vola = idio_vola
        self.factor_exposures = factor_exposures
        self.times = sorted(factor_covariance.keys())




def markowitz_problem(n_assets, n_factors):

    ### Variables
    asset_weights = cp.Variable(n_assets, name="asset_weights")
    cash_weight = cp.Variable(name="cash_weight")

    ### Data
    asset_weights_prev = cp.Parameter(n_assets, name="asset_weights_prev")
    cash_weight_prev = cp.Parameter(name="cash_weight_prev")
    factor_weights = cp.Parameter(n_factors, name="factor_weights")
    idio_mean = cp.Parameter(n_assets, name="idio_mean")
    factor_mean = cp.Parameter(n_factors, name="factor_mean")
    risk_free = cp.Parameter(name="risk_free")
    factor_covariance_chol = cp.Parameter((n_factors, n_factors), name="factor_covariance_chol")
    idio_vola = cp.Parameter(n_assets, name="idio_vola")
    exposures = cp.Parameter((n_assets, n_factors), name="exposures")
    kappa_short = cp.Parameter(n_assets, nonneg=True, name="kappa_short")   
    kappa_brw = cp.Parameter(nonneg=True, name="kappa_brw")
    kappa_bas = cp.Parameter(n_assets, nonneg=True, name="kappa_bas")
    kappa_mi = cp.Parameter(n_assets, nonneg=True, name="kappa_mi")
    

    ### Parameters
    assets_lower = cp.Parameter(n_assets, name="weights_lower")
    assets_upper = cp.Parameter(n_assets, name="weights_upper")
    cash_lower = cp.Parameter(name="cash_lower")
    cash_upper = cp.Parameter(name="cash_upper")
    trade_lower = cp.Parameter(n_assets, name="trade_lower")
    trade_upper = cp.Parameter(n_assets, name="trade_upper")
    turnover_limit = cp.Parameter(nonneg=True, name="turnover_limit")
    leverage_limit = cp.Parameter(nonneg=True, name="leverage_limit")
    rho_mean = cp.Parameter(nonneg=True, name="rho_mean")
    rho_covariance = cp.Parameter(nonneg=True, name="rho_covariance")
    gamma_hold = cp.Parameter(nonneg=True, name="gamma_hold")
    gamma_trade = cp.Parameter(nonneg=True, name="gamma_trade")
    gamma_turn = cp.Parameter(nonneg=True, name="gamma_turn")
    gamma_risk = cp.Parameter(nonneg=True, name="gamma_risk")
    risk_target = cp.Parameter(nonneg=True, name="risk_target")

    ### DPP enabling parameters
    variances = cp.sum(
        cp.multiply(exposures@factor_covariance_chol,
                    exposures@factor_covariance_chol
        ), axis=1
    ) + idio_vola**2
    volas_times_sqrt_rho = cp.CallbackParam(callback=lambda:
                                             variances.value**0.5 * rho_covariance.value**0.5, name="volas_times_sqrt_rho")
    _abs_weights = cp.Variable(n_assets, nonneg=True, name="_abs_weights")


    ### Return and risk
    return_wc = factor_mean@factor_weights + idio_mean@asset_weights + risk_free*cash_weight - rho_mean @ cp.abs(asset_weights)

    risk = cp.norm2(
        cp.hstack(
            [
                cp.norm2(factor_covariance_chol@factor_weights),
                cp.norm2(cp.multiply(idio_vola, asset_weights)),
            ]
        )
    )

    risk_wc = cp.norm2(
        cp.hstack(
            [
                risk, 
                volas_times_sqrt_rho@_abs_weights
                ]
        )
    )

    
    ### Costs
    holding_cost = kappa_short @ cp.pos(-asset_weights) + kappa_brw*cp.pos(-cash_weight)
    trading_cost = kappa_bas @ cp.abs(asset_weights - asset_weights_prev) + kappa_mi @ cp.abs(asset_weights - asset_weights_prev) + kappa_bas*cp.power(cash_weight - cash_weight_prev, 3/2)

    ### Turnover and leverage
    turnover = cp.norm1(asset_weights - asset_weights_prev) 
    leverage = cp.norm1(asset_weights)

    ### Objective
    obj = (
        return_wc
        - gamma_risk * cp.pos(risk_wc - risk_target)
        - gamma_hold * holding_cost
        - gamma_trade * trading_cost
        - gamma_turn * cp.pos(turnover - turnover_limit)
    )

    ### Constraints
    constraints = [
        leverage<=leverage_limit,
        asset_weights >= assets_lower,
        asset_weights <= assets_upper,
        cash_weight >= cash_lower,
        cash_weight <= cash_upper,
    ]








