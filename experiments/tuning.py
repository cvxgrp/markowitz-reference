import numpy as np
import pandas as pd
from markowitz import Data, Parameters
from dataclasses import dataclass
from collections import namedtuple
import multiprocessing as mp
import time
import cvxpy as cp

from backtest import (
    load_data,
    OptimizationInput,
    interest_and_fees,
    create_orders,
    execute_orders,
    Timing,
    BacktestResult,
)
from utils import synthetic_returns


@dataclass
class HyperParameters:
    gamma_hold: float
    gamma_trade: float
    gamma_turn: float
    gamma_leverage: float
    gamma_risk: float


@dataclass
class Targets:
    T_max: float
    L_max: float
    risk_target: float


@dataclass
class Limits:
    T_max: float
    L_max: float
    risk_max: float


def get_data_and_parameters(
    inputs: callable, hyperparameters: dataclass, targets: dataclass
):
    """
    param inputs: OptimizationInputs object
    param hyperparameters: HyperParameters object (gamma_hold, gamma_trade,
    gamma_turn, gamma_leverage, gamma_risk)
    param targets: Targets object (T_target, L_target, risk_target)
    """

    # TODO: correct??? We do not know the spread and volume of the last day,
    # right? It's a bit weird, becaus we assume knowledge of the price...
    latest_prices = inputs.prices.iloc[-1]
    portfolio_value = inputs.cash + inputs.quantities @ latest_prices

    spread_prediction = inputs.spread.iloc[-6:-1].mean().values

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

    if type(hyperparameters.gamma_hold) == pd.Series:
        gamma_hold = hyperparameters.gamma_hold.loc[t]
    else:
        gamma_hold = hyperparameters.gamma_hold
    if type(hyperparameters.gamma_trade) == pd.Series:
        gamma_trade = hyperparameters.gamma_trade.loc[t]
    else:
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
        # c_prev=(inputs.cash / portfolio_value),
        idio_mean=np.zeros(n_assets),
        factor_mean=inputs.mean.values,
        risk_free=inputs.risk_free,
        factor_covariance_chol=inputs.chol,
        idio_volas=np.zeros(n_assets),
        F=np.eye(n_assets),
        kappa_short=np.ones(n_assets) * 3 * (0.01) ** 2,  # 7.5% yearly
        kappa_borrow=inputs.risk_free,
        kappa_spread=np.ones(n_assets) * spread_prediction / 2,
        kappa_impact=np.zeros(n_assets),
    )

    param = Parameters(
        w_min=w_lower,
        w_max=w_upper,
        c_min=c_lower,
        c_max=c_upper,
        z_min=z_lower * np.ones(data.n_assets),
        z_max=z_upper * np.ones(data.n_assets),
        # T_target=targets.T_target,
        T_tar=targets.T_max,
        # L_target=targets.L_target,
        L_tar=targets.L_max,
        rho_mean=np.ones(n_assets) * rho_mean,
        rho_covariance=rho_covariance,
        gamma_hold=gamma_hold,
        gamma_trade=gamma_trade,
        gamma_turn=gamma_turn,
        gamma_risk=gamma_risk,
        risk_target=targets.risk_target,
        gamma_leverage=gamma_leverage,
    )

    return data, param


def markowitz_soft(
    data: Data,
    param: Parameters,
) -> tuple[np.ndarray, float, cp.Problem]:
    """
    Markowitz portfolio optimization.
    This function contains the code listing for the accompanying paper.
    """

    w, c = cp.Variable(data.n_assets), cp.Variable()

    z = w - data.w_prev
    T = cp.norm1(z)
    L = cp.norm1(w)

    # worst-case (robust) return
    factor_return = (data.F @ data.factor_mean).T @ w
    idio_return = data.idio_mean @ w
    mean_return = factor_return + idio_return + data.risk_free * c
    return_uncertainty = param.rho_mean @ cp.abs(w)
    return_wc = mean_return - return_uncertainty

    # asset volatilities
    factor_volas = cp.norm2(data.F @ data.factor_covariance_chol, axis=1)
    volas = factor_volas + data.idio_volas

    # portfolio risk
    factor_risk = cp.norm2((data.F @ data.factor_covariance_chol).T @ w)
    idio_risk = cp.norm2(cp.multiply(data.idio_volas, w))
    risk = cp.norm2(cp.hstack([factor_risk, idio_risk]))

    # worst-case (robust) risk
    risk_uncertainty = param.rho_covariance**0.5 * volas @ cp.abs(w)
    risk_wc = cp.norm2(cp.hstack([risk, risk_uncertainty]))

    asset_holding_cost = data.kappa_short @ cp.pos(-w)
    cash_holding_cost = data.kappa_borrow * cp.pos(-c)
    holding_cost = asset_holding_cost + cash_holding_cost

    spread_cost = data.kappa_spread @ cp.abs(z)
    impact_cost = data.kappa_impact @ cp.power(cp.abs(z), 3 / 2)
    trading_cost = spread_cost + impact_cost

    constraints = [
        cp.sum(w) + c == 1,
        # c == data.c_prev - cp.sum(z),
        param.c_min <= c,
        c <= param.c_max,
        param.w_min <= w,
        w <= param.w_max,
        param.z_min <= z,
        z <= param.z_max,
    ]

    objective = (
        return_wc
        - param.gamma_risk * cp.pos(risk_wc - param.risk_target)
        - param.gamma_hold * holding_cost
        - param.gamma_trade * trading_cost
        - param.gamma_turn * cp.pos(T - param.T_tar)
        - param.gamma_leverage * cp.pos(L - param.L_tar)
    )

    problem = cp.Problem(cp.Maximize(objective), constraints)

    try:
        problem.solve(verbose=False)
    except cp.SolverError:
        print("SolverError")
        print(problem.status)

    try:
        assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}, problem.status
    except AssertionError:
        print("Problem status: ", problem.status)

        return data.w_prev, data.c_prev, problem, False

    return w.value, c.value, problem, True


def markowitz_hard(
    data: Data, param: Parameters
) -> tuple[np.ndarray, float, cp.Problem]:
    """
    Markowitz portfolio optimization.
    This function contains the code listing for the accompanying paper.
    """

    w, c = cp.Variable(data.n_assets), cp.Variable()

    z = w - data.w_prev
    T = cp.norm1(z)
    L = cp.norm1(w)

    # worst-case (robust) return
    factor_return = (data.F @ data.factor_mean).T @ w
    idio_return = data.idio_mean @ w
    mean_return = factor_return + idio_return + data.risk_free * c
    return_uncertainty = param.rho_mean @ cp.abs(w)
    return_wc = mean_return - return_uncertainty

    # asset volatilities
    factor_volas = cp.norm2(data.F @ data.factor_covariance_chol, axis=1)
    volas = factor_volas + data.idio_volas

    # portfolio risk
    factor_risk = cp.norm2((data.F @ data.factor_covariance_chol).T @ w)
    idio_risk = cp.norm2(cp.multiply(data.idio_volas, w))
    risk = cp.norm2(cp.hstack([factor_risk, idio_risk]))

    # worst-case (robust) risk
    risk_uncertainty = param.rho_covariance**0.5 * volas @ cp.abs(w)
    risk_wc = cp.norm2(cp.hstack([risk, risk_uncertainty]))

    asset_holding_cost = data.kappa_short @ cp.pos(-w)
    cash_holding_cost = data.kappa_borrow * cp.pos(-c)
    holding_cost = asset_holding_cost + cash_holding_cost

    spread_cost = data.kappa_spread @ cp.abs(z)
    impact_cost = data.kappa_impact @ cp.power(cp.abs(z), 3 / 2)
    trading_cost = spread_cost + impact_cost

    objective = (
        return_wc - param.gamma_hold * holding_cost - param.gamma_trade * trading_cost
    )

    constraints = [
        cp.sum(w) + c == 1,
        # c == data.c_prev - cp.sum(z),
        param.c_min <= c,
        c <= param.c_max,
        param.w_min <= w,
        w <= param.w_max,
        param.z_min <= z,
        z <= param.z_max,
        L <= param.L_tar,
        T <= param.T_tar,
        risk_wc <= param.risk_target,
    ]

    # Naming the constraints
    constraints[0].name = "FullInvestment"
    # constraints[1].name = "Cash"
    constraints[1].name = "CLower"
    constraints[2].name = "CUpper"
    constraints[3].name = "WLower"
    constraints[4].name = "WUpper"
    constraints[5].name = "ZLower"
    constraints[6].name = "ZUpper"
    constraints[7].name = "Leverage"
    constraints[8].name = "Turnover"
    constraints[9].name = "Risk"

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve()
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}, problem.status
    return w.value, c.value, problem, True  # True means problem solved


def solve_hard_markowitz(inputs, hyperparamters, targets):
    """
    param inputs: OptimizationInputs object
    param hyperparameters: HyperParameters object (gamma_hold, gamma_trade,
    gamma_turn, gamma_leverage, gamma_risk)
    param targets: Targets object (T_target, L_target, risk_target)
    param limits: Limits object (T_max, L_max, risk_max)

    returns: w, c, problem, problem_solved
    """
    data, param = get_data_and_parameters(inputs, hyperparamters, targets)

    w, c, problem, problem_solved = markowitz_hard(data, param)

    return w, c, problem, problem_solved


def full_markowitz(inputs, hyperparamters, targets):
    """
    param inputs: OptimizationInputs object
    param hyperparameters: HyperParameters object (gamma_hold, gamma_trade,
    gamma_turn, gamma_leverage, gamma_risk)
    param targets: Targets object (T_target, L_target, risk_target)
    param limits: Limits object (T_max, L_max, risk_max)

    returns: w, c, problem, problem_solved
    """

    data, param = get_data_and_parameters(inputs, hyperparamters, targets)

    w, c, problem, problem_solved = markowitz_soft(data, param)
    return w, c, problem, problem_solved


def get_limits_and_targets(
    T_target,
    L_target,
    risk_target,
    T_max,
    L_max,
    risk_max,
):
    targets = Targets(
        T_target=T_target,
        L_target=L_target,
        risk_target=risk_target,
    )

    limits = Limits(
        T_max=T_max,
        L_max=L_max,
        risk_max=risk_max,
    )

    return targets, limits


def tune_parameters(
    strategy,
    prices,
    spread,
    volume,
    rf,
    train_len=None,
    verbose=False,
):
    """
    param strategy: strategy to tune
    param prices: prices
    param spread: spread
    param volume: volume
    param rf: risk free rate
    param train_len: length of training period; if None, all data is used for tuning
    """
    train_len = train_len or len(prices)
    test_len = len(prices) - 500 - train_len

    #### Helper functions ####

    def run_strategy(targets, hyperparameters):
        results = run_soft_backtest(
            strategy,
            targets=targets,
            hyperparameters=hyperparameters,
            prices=prices,
            spread=spread,
            volume=volume,
            rf=rf,
            verbose=False,
        )
        return results

    def sharpes(results):
        """
        returns sharpe ratio for training and test period"""
        returns_train = results.portfolio_returns.iloc[:-test_len]
        returns_test = results.portfolio_returns.iloc[-test_len:]

        sharpe_train = (
            np.sqrt(results.periods_per_year)
            * returns_train.mean()
            / returns_train.std()
        )
        sharpe_test = (
            np.sqrt(results.periods_per_year) * returns_test.mean() / returns_test.std()
        )

        return sharpe_train, sharpe_test

    def turnovers(results):
        """
        returns turnover for training and test period"""
        trades = results.quantities.diff()
        valuation_trades = (trades * prices).dropna()
        relative_trades = valuation_trades.div(results.portfolio_value, axis=0)
        relative_trades_train = relative_trades.iloc[:-test_len]
        relative_trades_test = relative_trades.iloc[-test_len:]

        turnover_train = (
            relative_trades_train.abs().sum(axis=1).mean() * results.periods_per_year
        )
        turnover_test = (
            relative_trades_test.abs().sum(axis=1).mean() * results.periods_per_year
        )

        return turnover_train, turnover_test

    def leverages(results):
        """
        returns leverage for training and test period"""
        leverage_train = results.asset_weights.abs().sum(axis=1).iloc[:-test_len].max()
        leverage_test = results.asset_weights.abs().sum(axis=1).iloc[-test_len:].max()
        return leverage_train, leverage_test

    def risks(results):
        """
        returns risk for training and test period"""
        returns_train = results.portfolio_returns.iloc[:-test_len]
        returns_test = results.portfolio_returns.iloc[-test_len:]
        risk_train = returns_train.std() * np.sqrt(results.periods_per_year)
        risk_test = returns_test.std() * np.sqrt(results.periods_per_year)
        return risk_train, risk_test

    def accept_new_parameter(results, sharpe_old):
        sharpe_train, _ = sharpes(results)

        if sharpe_train <= sharpe_old:
            return False
        else:
            # check turnover
            turnover_train, turnover_test = turnovers(results)
            if verbose:
                print(f"Turnover: {turnover_train}, {turnover_test}")
            if turnover_train > 100:
                return False
            # check leverage
            leverage_train, leverage_test = leverages(results)
            if verbose:
                print(f"Leverage: {leverage_train}, {leverage_test}")
            if leverage_train > 2:
                return False
            # check risk
            risk_train, risk_test = risks(results)
            if verbose:
                print(f"Risk: {risk_train}, {risk_test}")
            if risk_train > 0.15:
                return False

        return True

    #### Main ####
    parameters_to_results = {}

    # Initial hyperparameters
    # gamma_turns = 2.5e-3
    # gamma_leverages = 5e-4
    # gamma_risks = 5e-2
    # hyperparameters = HyperParameters(1,1, 1e-3, 2.5e-9, 2.5e-2)

    gamma_turns = 1e-3
    gamma_leverages = 2.5e-9
    gamma_risks = 2.5e-2

    hyperparameter_list = [1, 1, gamma_turns, gamma_leverages, gamma_risks]
    hyperparameters = HyperParameters(*hyperparameter_list)

    # Set soft targets and limits
    targets = Targets(
        T_max=50 / 252,
        L_max=1.6,
        risk_target=0.1 / np.sqrt(252),
    )

    # Initial soft solve
    backtest = 1  # keep track of number of backtests
    results = run_strategy(targets, hyperparameters)
    results_best = results
    parameters_to_results[backtest] = (hyperparameters, results, results_best)

    sharpe_train, sharpe_test = sharpes(results)
    if verbose:
        print(f"Initial sharpes: {sharpe_train}, {sharpe_test}")
    sharpe_train_old = sharpe_train
    sharpe_test_old = sharpe_test

    best_backtest = 1
    non_inprove_in_a_row = 0  # stop when this is n_params-1

    n_params = 5
    iteration = 1  # keep track of number of iterations

    while non_inprove_in_a_row < n_params - 1:
        update_var = iteration % n_params - 1  # update one parameter at a time
        gamma_temp = hyperparameter_list[update_var] * 1.25  # increase by 25%

        hyperparameter_list_temp = hyperparameter_list.copy()
        hyperparameter_list_temp[update_var] = gamma_temp

        hyperparameters_temp = HyperParameters(*hyperparameter_list_temp)

        results = run_strategy(targets, hyperparameters_temp)
        backtest += 1
        sharpe_train, sharpe_test = sharpes(results)

        if accept_new_parameter(results, sharpe_train_old):
            # update hyperparameters
            hyperparameter_list = hyperparameter_list_temp.copy()
            hyperparameters = hyperparameters_temp

            # update results
            best_backtest = backtest
            results_best = results
            sharpe_train_old = sharpe_train
            sharpe_test_old = sharpe_test
            parameters_to_results[backtest] = (hyperparameters, results, results_best)

            # reset non_inprove_in_a_row
            non_inprove_in_a_row = 0
        else:
            parameters_to_results[backtest] = (hyperparameters, results, results_best)

            gamma_temp = hyperparameter_list[update_var] * 0.8  # decrease by 20%
            hyperparameter_list_temp[update_var] = gamma_temp
            hyperparameters_temp = HyperParameters(*hyperparameter_list_temp)

            results = run_strategy(targets, hyperparameters_temp)
            backtest += 1
            sharpe_train, sharpe_test = sharpes(results)

            if accept_new_parameter(results, sharpe_train_old):
                # update hyperparameters
                hyperparameter_list = hyperparameter_list_temp.copy()
                hyperparameters = hyperparameters_temp

                # update results
                best_backtest = backtest
                results_best = results
                sharpe_train_old = sharpe_train
                sharpe_test_old = sharpe_test
                parameters_to_results[backtest] = (
                    hyperparameters,
                    results,
                    results_best,
                )

                # reset non_inprove_in_a_row
                non_inprove_in_a_row = 0
            else:
                parameters_to_results[backtest] = (
                    hyperparameters,
                    results,
                    results_best,
                )
                non_inprove_in_a_row += 1
                if verbose:
                    print("In a row: " + str(non_inprove_in_a_row))

        # parameters_to_results[iteration] = (hyperparameters, results)
        if verbose:
            print(
                f"\nIteration number {iteration};\
                      current sharpes:\
                          {sharpe_train_old, sharpe_test_old}"
            )
            print(f"Hyperparameters: {hyperparameters}")
        iteration += 1

    print("Done!")

    return parameters_to_results, best_backtest


def tune_in_parallel(
    all_prices,
    all_spreads,
    all_volumes,
    all_rfs,
):
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(
        tune_parameters,
        zip(
            [full_markowitz] * len(all_prices),
            all_prices,
            all_spreads,
            all_volumes,
            all_rfs,
        ),
    )
    pool.close()
    pool.join()

    return results


def run_hard_backtest(
    strategy: callable,
    targets: namedtuple,
    hyperparameters: namedtuple,
    prices=None,
    spread=None,
    volume=None,
    rf=None,
    verbose: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Run a simplified backtest for a given strategy.
    At time t we use data from t-lookback to t to compute the optimal portfolio
    weights and then execute the trades at time t.
    """

    if prices is None:
        train_len = 1250
        prices, spread, rf, volume = load_data()
        prices, spread, rf, volume = (
            prices.iloc[train_len:],
            spread.iloc[train_len:],
            rf.iloc[train_len:],
            volume.iloc[train_len:],
        )

    n_assets = prices.shape[1]

    lookback = 500
    forward_smoothing = 5

    constraint_names = [
        "FullInvestment",
        # "Cash",
        "CLower",
        "CUpper",
        "WLower",
        "WUpper",
        "ZLower",
        "ZUpper",
        "Leverage",
        "Turnover",
        "Risk",
    ]

    quantities = np.zeros(n_assets)
    cash = 1e6

    post_trade_cash = []
    post_trade_quantities = []
    timings = []
    dual_optimals = (
        pd.DataFrame(
            columns=constraint_names,
            index=prices.index[lookback:-1],
        )
        * np.nan
    )

    returns = prices.pct_change().dropna()
    means = (
        synthetic_returns(
            prices, information_ratio=0.07, forward_smoothing=forward_smoothing
        )
        .shift(-1)
        .dropna()
    )  # At time t includes data up to t+1
    covariance_df = returns.ewm(halflife=125).cov()  # At time t includes data up to t
    indices = range(lookback, len(prices) - forward_smoothing)
    days = [prices.index[t] for t in indices]
    covariances = {}
    cholesky_factorizations = {}
    for day in days:
        covariances[day] = covariance_df.loc[day]
        cholesky_factorizations[day] = np.linalg.cholesky(covariances[day].values)

    for t in range(lookback, len(prices) - forward_smoothing):
        start_time = time.perf_counter()

        day = prices.index[t]

        if verbose:
            print(f"Day {t} of {len(prices)-forward_smoothing}, {day}")

        prices_t = prices.iloc[t - lookback : t + 1]  # Up to t
        spread_t = spread.iloc[t - lookback : t + 1]

        mean_t = means.loc[day]  # Forecast for return t to t+1
        covariance_t = covariances[day]  # Forecast for covariance t to t+1
        chol_t = cholesky_factorizations[day]
        volas_t = np.sqrt(np.diag(covariance_t.values))

        inputs_t = OptimizationInput(
            prices_t,
            mean_t,
            chol_t,
            volas_t,
            spread_t,
            quantities,
            cash,
            targets.risk_target,
            rf.iloc[t],
        )

        w, _, problem, problem_solved = strategy(
            inputs_t,
            hyperparameters,
            targets=targets,
        )

        latest_prices = prices.iloc[t]  # At t
        latest_spread = spread.iloc[t]

        cash += interest_and_fees(
            cash, rf.iloc[t - 1], quantities, prices.iloc[t - 1], day
        )
        trade_quantities = create_orders(w, quantities, cash, latest_prices)
        quantities += trade_quantities
        cash += execute_orders(
            latest_prices,
            trade_quantities,
            latest_spread,
        )

        post_trade_cash.append(cash)
        post_trade_quantities.append(quantities.copy())

        if problem_solved:
            for name in constraint_names:
                dual_optimals.loc[day, name] = problem.constraints[
                    constraint_names.index(name)
                ].dual_value

        # Timings
        end_time = time.perf_counter()
        timings.append(Timing.get_timing(start_time, end_time, problem))

    post_trade_cash = pd.Series(
        post_trade_cash, index=prices.index[lookback:-forward_smoothing]
    )
    post_trade_quantities = pd.DataFrame(
        post_trade_quantities,
        index=prices.index[lookback:-forward_smoothing],
        columns=prices.columns,
    )

    return BacktestResult(
        post_trade_cash,
        post_trade_quantities,
        targets.risk_target,
        timings,
        dual_optimals,
    )


def run_soft_backtest(
    strategy: callable,
    targets: namedtuple,
    hyperparameters: namedtuple,
    prices=None,
    spread=None,
    volume=None,
    rf=None,
    verbose: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Run a simplified backtest for a given strategy.
    At time t we use data from t-lookback to t to compute the optimal portfolio
    weights and then execute the trades at time t.

    param strategy: a function that takes an OptimizationInput and returns
    w, c, problem, problem_solved
    param prices: a DataFrame of prices
    param spread: a DataFrame of bid-ask spreads
    param volume: a DataFrame of trading volumes
    param rf: a Series of risk-free rates
    param targets: a namedtuple of targets (T_target, L_target, risk_target)
    param limits: a namedtuple of limits (T_max, L_max, risk_max)
    param hyperparameters: a namedtuple of hyperparameters (gamma_hold,
    gamma_trade, gamma_turn, gamma_risk, gamma_leverage)
    param verbose: whether to print progress

    return: tuple of BacktestResult instance and DataFrame of dual optimal values
    """

    if prices is None:
        train_len = 1250
        prices, spread, rf, volume = load_data()
        prices, spread, rf, volume = (
            prices.iloc[train_len:],
            spread.iloc[train_len:],
            rf.iloc[train_len:],
            volume.iloc[train_len:],
        )

    n_assets = prices.shape[1]
    lookback = 500
    forward_smoothing = 5

    timings = []

    returns = prices.pct_change().dropna()
    means = (
        synthetic_returns(
            prices, information_ratio=0.07, forward_smoothing=forward_smoothing
        )
        .shift(-1)
        .dropna()
    )  # At time t includes data up to t+1
    covariance_df = returns.ewm(halflife=125).cov()  # At time t includes data up to t
    indices = range(lookback, len(prices) - forward_smoothing)
    days = [prices.index[t] for t in indices]
    covariances = {}
    cholesky_factorizations = {}
    for day in days:
        covariances[day] = covariance_df.loc[day]
        cholesky_factorizations[day] = np.linalg.cholesky(covariances[day].values)

    quantities = np.zeros(n_assets)
    cash = 1e6

    # To store results
    post_trade_cash = []
    post_trade_quantities = []
    # dual_optimals = (
    #     pd.DataFrame(
    #         columns=constraint_names,
    #         index=prices.index[lookback:-1],
    #     )
    #     * np.nan
    # )

    for t in range(lookback, len(prices) - forward_smoothing):
        start_time = time.perf_counter()
        day = prices.index[t]

        if verbose and t % 100 == 0:
            print(f"Day {t} of {len(prices)-1}, {day}")

        prices_t = prices.iloc[t - lookback : t + 1]  # Up to t
        spread_t = spread.iloc[t - lookback : t + 1]
        volume.iloc[t - lookback : t + 1]

        mean_t = means.loc[day]  # Forecast for return t to t+1
        covariance_t = covariances[day]  # Forecast for covariance t to t+1
        chol_t = cholesky_factorizations[day]
        volas_t = np.sqrt(np.diag(covariance_t.values))

        inputs_t = OptimizationInput(
            prices_t,
            mean_t,
            chol_t,
            volas_t,
            spread_t,
            quantities,
            cash,
            targets.risk_target,
            rf.iloc[t],
        )

        w, _, problem, problem_solved = strategy(
            inputs_t,
            hyperparameters,
            targets=targets,
        )

        latest_prices = prices.iloc[t]  # At t
        latest_spread = spread.iloc[t]

        cash += interest_and_fees(
            cash, rf.iloc[t - 1], quantities, prices.iloc[t - 1], day
        )
        trade_quantities = create_orders(w, quantities, cash, latest_prices)
        quantities += trade_quantities
        cash += execute_orders(
            latest_prices,
            trade_quantities,
            latest_spread,
        )

        # print(((np.abs(trade_quantities)*latest_prices)/volume.iloc[t]).mean())

        post_trade_cash.append(cash)
        post_trade_quantities.append(quantities.copy())

        # if problem_solved:
        #     for name in constraint_names:
        #         dual_optimals.loc[day, name] = problem.constraints[
        #             constraint_names.index(name)
        #         ].dual_value

        # Timings
        end_time = time.perf_counter()
        if problem_solved:
            timings.append(Timing.get_timing(start_time, end_time, problem))
        else:
            timings.append(None)

    post_trade_cash = pd.Series(
        post_trade_cash, index=prices.index[lookback:-forward_smoothing]
    )
    post_trade_quantities = pd.DataFrame(
        post_trade_quantities,
        index=prices.index[lookback:-forward_smoothing],
        columns=prices.columns,
    )

    return BacktestResult(
        post_trade_cash, post_trade_quantities, targets.risk_target, timings
    )


def main():
    gamma_risk = 0.05
    gamma_turn = 0.0025
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

    results, _ = run_soft_backtest(
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
