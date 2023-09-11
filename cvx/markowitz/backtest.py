import pandas as pd
import cvxpy as cp
from dataclasses import dataclass
from collections import namedtuple

ParameterizedProblem = namedtuple("ParameterizedProblem", ["problem", "variables", "parameters"])

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

class Data:
    def __init__(self, asset_weights_prev, cash_weight_prev, idio_mean, factor_mean, risk_free, factor_covariance, idio_vola, exposures, kappa_short, kappa_brw, kappa_bas, kappa_mi):
        """
        param asset_weights_prev: (n_assets,) array of previous asset weights
        param cash_weight_prev: previous cash weight
        param idio_mean: (n_assets,) array of idiosyncratic mean returns
        param factor_mean: (n_factors,) array of factor mean returns
        param risk_free: risk free rate
        param factor_covariance: (n_factors, n_factors) covariance matrix of
            factor returns
        param idio_vola: (n_assets,) array of idiosyncratic volatilities
        param exposures: (n_assets, n_factors) array of factor exposures
        param kappa_short: (n_assets,) array of shorting cost
        param kappa_brw: borrowing cost
        param kappa_bas: (n_assets,) array of bid ask spreads
        param kappa_mi: (n_assets,) array of market impact costs          
        """

        self.asset_weights_prev = asset_weights_prev
        self.cash_weight_prev = cash_weight_prev
        self.idio_mean = idio_mean
        self.factor_mean = factor_mean
        self.risk_free = risk_free
        self.factor_covariance = factor_covariance
        self.idio_vola = idio_vola
        self.exposures = exposures
        self.kappa_short = kappa_short
        self.kappa_brw = kappa_brw
        self.kappa_bas = kappa_bas
        self.kappa_mi = kappa_mi

class Parameters:
    def __init__(self, assets_lower, assets_upper, cash_lower, cash_upper, trade_lower, trade_upper, turnover_limit, leverage_limit, rho_mean, rho_covariance, gamma_hold, gamma_trade, gamma_turn, gamma_risk, risk_target):
        """
        param assets_lower: (n_assets,) array of lower bounds on asset weights
        param assets_upper: (n_assets,) array of upper bounds on asset weights
        param cash_lower: lower bound on cash weight
        param cash_upper: upper bound on cash weight
        param trade_lower: (n_assets,) array of lower bounds on trades
        param trade_upper: (n_assets,) array of upper bounds on trades
        param turnover_limit: turnover limit
        param leverage_limit: leverage limit
        param rho_mean: (n_assets,) array of mean returns for rho
        param rho_covariance: covariance of rho
        param gamma_hold: holding cost
        param gamma_trade: trading cost
        param gamma_turn: turnover cost
        param gamma_risk: risk cost
        param risk_target: risk target
        """
        self.parameters = {
            "assets_lower": assets_lower, 
            "assets_upper": assets_upper,
            "cash_lower": cash_lower,
            "cash_upper": cash_upper,
            "trade_lower": trade_lower,
            "trade_upper": trade_upper,
            "turnover_limit": turnover_limit,
            "leverage_limit": leverage_limit,
            "rho_mean": rho_mean,
            "rho_covariance": rho_covariance,
            "gamma_hold": gamma_hold,
            "gamma_trade": gamma_trade,
            "gamma_turn": gamma_turn,
            "gamma_risk": gamma_risk,
            "risk_target": risk_target
            }



        self.assets_lower = assets_lower
        self.assets_upper = assets_upper
        self.cash_lower = cash_lower
        self.cash_upper = cash_upper
        self.trade_lower = trade_lower
        self.trade_upper = trade_upper
        self.turnover_limit = turnover_limit
        self.leverage_limit = leverage_limit
        self.rho_mean = rho_mean
        self.rho_covariance = rho_covariance
        self.gamma_hold = gamma_hold
        self.gamma_trade = gamma_trade
        self.gamma_turn = gamma_turn
        self.gamma_risk = gamma_risk
        self.risk_target = risk_target



class Markowitz:

    def __init__(self, n_assets, n_factors) -> None:
        
        self.parameterized_problem = self._build(n_assets, n_factors)

    @staticmethod
    def _build(n_assets, n_factors):

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
        rho_mean = cp.Parameter(n_assets, nonneg=True, name="rho_mean")
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
                                                variances.value**0.5 * rho_covariance.value**0.5,
                                                 shape=variances.shape, 
                                                 nonneg=True,
                                                 name="volas_times_sqrt_rho")
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
        trades = asset_weights - asset_weights_prev
        holding_cost = kappa_short @ cp.pos(-asset_weights) + kappa_brw*cp.pos(-cash_weight)
        trading_cost = kappa_bas @ cp.abs(trades) + kappa_mi@cp.power(trades, 3/2)

        ### Turnover and leverage
        turnover = cp.norm1(trades) 
        leverage = cp.norm1(asset_weights)

        ### Objective
        objective = (
            return_wc
            - gamma_risk * cp.pos(risk_wc - risk_target)
            - gamma_hold * holding_cost
            - gamma_trade * trading_cost
            - gamma_turn * cp.pos(turnover - turnover_limit)
        )

        ### Constraints
        constraints = [
            factor_weights==exposures.T@asset_weights,
            cp.sum(asset_weights) + cash_weight == 1,
            cash_weight == cash_weight_prev - cp.sum(trades),
            cash_weight >= cash_lower,
            cash_weight <= cash_upper,
            leverage<=leverage_limit,
            asset_weights >= assets_lower,
            asset_weights <= assets_upper,
            cash_weight >= cash_lower,
            cash_weight <= cash_upper,
            trades >= trade_lower,
            trades <= trade_upper,
            _abs_weights >= cp.abs(asset_weights),
        ]

        ### Problem
        problem = cp.Problem(cp.Maximize(objective), constraints)

        return ParameterizedProblem(
            problem=problem,
            variables=problem.var_dict,
            parameters=problem.param_dict,
        )
    
    def backtest(self, data, parameters):
        for parameter in parameters.keys():
            self.parameterized_problem.parameters[parameter].value = parameters[parameter]

        










