import numpy as np
import pandas as pd
from markowitz import Data, Parameters, markowitz
from dataclasses import dataclass
import multiprocessing as mp

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
        # T_target=targets.T_target,
        T_max=targets.T_max,
        # L_target=targets.L_target,
        L_max=targets.L_max,
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

    w, c, problem, problem_solved = markowitz(data, param)
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

    #### Helper functions ####

    def run_strategy(targets, hyperparameters):
        results = run_markowitz(
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
        returns_train = results.portfolio_returns.iloc[:train_len]
        returns_test = results.portfolio_returns.iloc[train_len:]
        sharpe_train = np.sqrt(252) * returns_train.mean() / returns_train.std()
        sharpe_test = np.sqrt(252) * returns_test.mean() / returns_test.std()

        return sharpe_train, sharpe_test

    def turnovers(results):
        """
        returns turnover for training and test period"""
        trades = results.quantities.diff()
        valuation_trades = (trades * prices).dropna()
        relative_trades = valuation_trades.div(results.portfolio_value, axis=0)
        relative_trades_train = relative_trades.iloc[:train_len]
        relative_trades_test = relative_trades.iloc[train_len:]

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
        leverage_train = results.asset_weights.abs().sum(axis=1).iloc[:train_len].max()
        leverage_test = results.asset_weights.abs().sum(axis=1).iloc[train_len:].max()
        return leverage_train, leverage_test

    def risks(results):
        """
        returns risk for training and test period"""
        returns_train = results.portfolio_returns.iloc[:train_len]
        returns_test = results.portfolio_returns.iloc[train_len:]
        risk_train = returns_train.std() * np.sqrt(252)
        risk_test = returns_test.std() * np.sqrt(252)
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
    gamma_turns = 2.5e-3
    gamma_leverages = 5e-4
    gamma_risks = 5e-2

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
