import pandas as pd
import cvxpy as cp




def markowitz_problem(n_assets, n_factors):

    ### Variables
    asset_weights = cp.Variable(n_assets)
    cash_weight = cp.Variable()

    ### Data
    asset_weights_prev = cp.Parameter(n_assets)
    cash_weight_prev = cp.Parameter()
    factor_weights = cp.Parameter(n_factors)
    exposures = cp.Parameter((n_assets, n_factors))

    ### Parameters
    asset_mean = cp.Parameter(n_assets)
    factor_covariance_chol = cp.Parameter((n_factors, n_factors))
    idiosyncratic_vola = cp.Parameter(n_assets)

    rho_mean = cp.Parameter(nonneg=True)
    rho_covariance = cp.Parameter(nonneg=True)

    gamma_hold = cp.Parameter(nonneg=True)
    gamma_trade = cp.Parameter(nonneg=True)
    gamma_turn = cp.Parameter(nonneg=True)
    gamma_risk = cp.Parameter(nonneg=True)

    ### DPP enabling parameters
    variances = cp.sum(
        cp.multiply(exposures@factor_covariance_chol,
                    exposures@factor_covariance_chol
        ), axis=1
    ) + idiosyncratic_vola**2
    volas_times_sqrt_rho = cp.CallbackParam(callback=lambda:
                                             variances.value**0.5 * rho_covariance.value**0.5)

    risk = cp.norms(
        cp.hstac(
            [
                cp.norm2(factor_covariance_chol@factor_weights),
                cp.norm2(cp.multiply(idiosyncratic_vola, asset_weights)),
            ]
        )
    )






    



def mean_var_factor(
n, k, sigma_tar, lower=-0.3, upper=0.4, leverage=2, lower_cash=None, upper_cash=None
):
    """
    Mean variance optimization with factor model
    """

    w = cp.Variable(n + 1, name="w")  # Last element is cash
    f = cp.Variable(k, name="f")

    w_old = cp.Parameter(n + 1, name="w_old")

    alpha_w = cp.Parameter(n, name="alpha_w")
    alpha_f = cp.Parameter(k, name="alpha_f")
    exposure = cp.Parameter((n, k), name="exposure")
    chol_f = cp.Parameter((k, k), name="chol_f")
    idio_vola = cp.Parameter(n, name="idio_vola")
    trading_costs = cp.Parameter(n, name="trading_costs")
    trading_costs_times_w_old = cp.Parameter(n, name="trading_costs_times_w_old")

    # if transaction_costs:
    ret = (
        alpha_f @ f
        + alpha_w @ w[:-1]
        - cp.norm1(cp.multiply(trading_costs, w[:-1]) - trading_costs_times_w_old)
    )
    # else:
    #     ret = alpha_f @ f + alpha_w @ w[:-1]

    obj = cp.Maximize(ret)

    risk = cp.norm2(cp.hstack([cp.norm2(chol_f.T @ f), cp.norm2(cp.multiply(idio_vola, w[:-1]))]))

    cons = [cp.sum(w) == 1, risk <= sigma_tar / np.sqrt(250), f == exposure.T @ w[:-1]]

    if lower is not None:
        cons.append(w[:-1] >= lower)
    if upper is not None:
        cons.append(w[:-1] <= upper)
    if leverage is not None:
        cons.append(cp.norm1(w[:-1]) <= leverage)

    if lower_cash is not None:
        cons.append(w[-1] >= lower_cash)
    if upper_cash is not None:
        cons.append(w[-1] <= upper_cash)

    prob = cp.Problem(obj, cons)

    return (
        prob,
        w,
        f,
        alpha_w,
        alpha_f,
        exposure,
        chol_f,
        idio_vola,
        trading_costs,
        trading_costs_times_w_old,
        w_old,
    )