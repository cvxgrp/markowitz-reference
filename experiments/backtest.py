from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import pickle
import sys
import time
from typing import Callable
import numpy as np
import cvxpy as cp
import pandas as pd
from utils import synthetic_returns
from collections import namedtuple

# hack to allow importing from parent directory without having a package
sys.path.append(str(Path(__file__).parent.parent))


def data_folder():
    return Path(__file__).parent.parent / "data"


@lru_cache(maxsize=1)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(data_folder() / "prices.csv", index_col=0, parse_dates=True)
    spread = pd.read_csv(data_folder() / "spreads.csv", index_col=0, parse_dates=True)
    volume = pd.read_csv(
        data_folder() / "volumes_shares.csv", index_col=0, parse_dates=True
    )
    rf = pd.read_csv(data_folder() / "rf.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return prices, spread, volume, rf


@dataclass
class OptimizationInput:
    """
    At time t, we have data from t-lookback to t-1.
    """

    prices: pd.DataFrame
    mean: pd.Series
    covariance: pd.DataFrame
    spread: pd.DataFrame
    volume: pd.DataFrame
    quantities: np.ndarray
    cash: float
    risk_target: float
    risk_free: float

    @property
    def n_assets(self) -> int:
        return self.prices.shape[1]


def run_backtest(
    strategy: Callable, risk_target: float, verbose: bool = False
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Run a simplified backtest for a given strategy.
    At time t we use data from t-lookback to t to compute the optimal portfolio
    weights and then execute the trades at time t.
    """

    prices, spread, volume, rf = load_data()
    n_assets = prices.shape[1]

    lookback = 500
    forward_smoothing = 5

    quantities = np.zeros(n_assets)
    cash = 1e6

    post_trade_cash = []
    post_trade_quantities = []
    timings = []

    returns = prices.pct_change().dropna()
    means = (
        synthetic_returns(
            prices, information_ratio=0.15, forward_smoothing=forward_smoothing
        )
        .shift(-1)
        .dropna()
    )  # At time t includes data up to t+1
    covariance_df = returns.ewm(halflife=125).cov()  # At time t includes data up to t
    days = returns.index
    covariances = {}
    for day in days:
        covariances[day] = covariance_df.loc[day]

    for t in range(lookback, len(prices) - forward_smoothing):
        start_time = time.perf_counter()

        day = prices.index[t]

        if verbose:
            print(f"Day {t} of {len(prices)-forward_smoothing}, {day}")

        prices_t = prices.iloc[t - lookback : t + 1]  # Up to t
        spread_t = spread.iloc[t - lookback : t + 1]
        volume_t = volume.iloc[t - lookback : t + 1]

        mean_t = means.loc[day]  # Forecast for return t to t+1
        covariance_t = covariances[day]  # Forecast for covariance t to t+1

        inputs_t = OptimizationInput(
            prices_t,
            mean_t,
            covariance_t,
            spread_t,
            volume_t,
            quantities,
            cash,
            risk_target,
            rf.iloc[t],
        )

        w, _, problem = strategy(inputs_t)

        latest_prices = prices.iloc[t]  # At t
        latest_spread = spread.iloc[t]

        cash += interest_and_fees(
            cash, rf.iloc[t - 1], quantities, prices.iloc[t - 1], day
        )
        trade_quantities = create_orders(w, quantities, cash, latest_prices)
        quantities += trade_quantities
        cash += execute_orders(latest_prices, trade_quantities, latest_spread)

        post_trade_cash.append(cash)
        post_trade_quantities.append(quantities.copy())

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

    return BacktestResult(post_trade_cash, post_trade_quantities, risk_target, timings)


def run_markowitz(
    strategy: callable,
    targets: namedtuple,
    limits: namedtuple,
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
        prices, spread, volume, rf = load_data()

    n_assets = prices.shape[1]
    lookback = 500
    forward_smoothing = 5

    constraint_names = [
        "FullInvestment",
        "Cash",
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

    timings = []

    returns = prices.pct_change().dropna()
    means = (
        synthetic_returns(
            prices, information_ratio=0.15, forward_smoothing=forward_smoothing
        )
        .shift(-1)
        .dropna()
    )  # At time t includes data up to t+1
    covariance_df = returns.ewm(halflife=125).cov()  # At time t includes data up to t
    days = returns.index
    covariances = {}
    for day in days:
        covariances[day] = covariance_df.loc[day]

    quantities = np.zeros(n_assets)
    cash = 10 * 1e6

    # To store results
    post_trade_cash = []
    post_trade_quantities = []
    dual_optimals = (
        pd.DataFrame(
            columns=constraint_names,
            index=prices.index[lookback:-1],
        )
        * np.nan
    )

    for t in range(lookback, len(prices) - forward_smoothing):
        start_time = time.perf_counter()
        day = prices.index[t]

        if verbose and t % 100 == 0:
            print(f"Day {t} of {len(prices)-1}, {day}")

        prices_t = prices.iloc[t - lookback : t + 1]  # Up to t
        spread_t = spread.iloc[t - lookback : t + 1]
        volume_t = volume.iloc[t - lookback : t + 1]

        mean_t = means.loc[day]  # Forecast for return t to t+1
        covariance_t = covariances[day]  # Forecast for covariance t to t+1

        inputs_t = OptimizationInput(
            prices_t,
            mean_t,
            covariance_t,
            spread_t,
            volume_t,
            quantities,
            cash,
            limits.risk_max,
            rf.iloc[t],
        )

        w, _, problem, problem_solved = strategy(
            inputs_t, hyperparameters, targets=targets, limits=limits
        )

        latest_prices = prices.iloc[t]  # At t
        # portfolio_value = cash + quantities @ latest_prices
        latest_spread = spread.iloc[t]
        # market_volume = volume.iloc[t]  # / portfolio_value

        # print(1, latest_volume.max())
        # print(2, latest_volume.min())
        # latest_volas = np.diag(covariance_t) ** 0.5

        cash += interest_and_fees(
            cash, rf.iloc[t - 1], quantities, prices.iloc[t - 1], day
        )
        trade_quantities = create_orders(w, quantities, cash, latest_prices)
        quantities += trade_quantities
        cash += execute_orders(
            latest_prices,
            trade_quantities,
            latest_spread,
            # market_volume,
            # latest_volas,
            # portfolio_value,
        )

        # print(((np.abs(trade_quantities)*latest_prices)/volume.iloc[t]).mean())

        post_trade_cash.append(cash)
        post_trade_quantities.append(quantities.copy())

        if problem_solved:
            for name in constraint_names:
                dual_optimals.loc[day, name] = problem.constraints[
                    constraint_names.index(name)
                ].dual_value

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

    return (
        BacktestResult(
            post_trade_cash, post_trade_quantities, limits.risk_max, timings
        ),
        dual_optimals,
    )


def create_orders(w, quantities, cash, latest_prices) -> np.array:
    portfolio_value = cash + quantities @ latest_prices
    w_prev = (quantities * latest_prices) / portfolio_value

    z = w - w_prev
    trades = z * portfolio_value
    trade_quantities = trades / latest_prices
    return trade_quantities.values


def execute_orders(
    latest_prices,
    trade_quantities,
    latest_spread,
    # market_volume,
    # latest_volas,
) -> float:
    sell_order_quantities = np.clip(trade_quantities, None, 0)
    buy_order_quantities = np.clip(trade_quantities, 0, None)

    sell_order_prices = latest_prices * (1 - latest_spread / 2)
    buy_order_prices = latest_prices * (1 + latest_spread / 2)

    sell_receipt = -sell_order_quantities @ sell_order_prices
    buy_payment = buy_order_quantities @ buy_order_prices

    # # market impact cost TODO: correct???
    # kappa_impact = latest_volas / (market_volume**0.5)
    # # print(market_volume.max())
    # market_cost = kappa_impact @ (
    #     np.abs(trade_quantities * latest_prices) ** (3 / 2)
    # )  # * portfolio_value

    # print(market_cost)
    # print(2, market_cost / portfolio_value / (0.01**2))

    return sell_receipt - buy_payment  # - market_cost


def interest_and_fees(
    cash: float, rf: float, quantities: pd.Series, prices: pd.Series, day: pd.Timestamp
) -> float:
    """
    From t-1 to t we either earn interest on cash or pay interest on borrowed cash.
    We also pay a fee for shorting (stark simplification: using the same rate).

    cash: cash at t-1
    rf: risk free rate from t-1 to t
    quantities: quantities at t-1
    prices: prices at t-1
    day: day t
    Note on rf: the Effective Federal Funds Rate uses ACT/360.
    """
    days_t_to_t_minus_1 = (day - prices.name).days
    cash_interest = cash * (1 + rf) ** days_t_to_t_minus_1 - cash
    short_valuations = np.clip(quantities, None, 0) * prices
    short_value = short_valuations.sum()
    shorting_fee = short_value * (1 + rf) ** days_t_to_t_minus_1 - short_value
    return cash_interest + shorting_fee


@dataclass
class Timing:
    solver: float
    cvxpy: float
    other: float

    @property
    def total(self):
        return self.solver + self.cvxpy + self.other

    @classmethod
    def get_timing(
        cls, start_time: float, end_time: float, problem: cp.Problem | None
    ) -> Timing:
        if problem:
            solver_time = problem.solver_stats.solve_time
            cvxpy_time = problem.compilation_time
            other_time = end_time - start_time - solver_time - cvxpy_time
            return cls(solver_time, cvxpy_time, other_time)
        else:
            return cls(0, 0, 0)


@dataclass
class BacktestResult:
    cash: pd.Series
    quantities: pd.DataFrame
    risk_target: float
    timings: list[Timing]

    @property
    def valuations(self) -> pd.DataFrame:
        prices = load_data()[0].loc[self.history]
        return self.quantities * prices

    @property
    def portfolio_value(self) -> pd.Series:
        return self.cash + self.valuations.sum(axis=1)

    @property
    def portfolio_returns(self):
        return self.portfolio_value.pct_change().dropna()

    @property
    def periods_per_year(self):
        return len(self.history) / ((self.history[-1] - self.history[0]).days / 365.25)

    @property
    def history(self):
        return self.cash.index

    @property
    def cash_weight(self):
        return self.cash / self.portfolio_value

    @property
    def asset_weights(self):
        return self.valuations.div(self.portfolio_value, axis=0)

    @property
    def turnover(self) -> float:
        trades = self.quantities.diff()
        prices = load_data()[0].loc[self.history]
        valuation_trades = trades * prices
        relative_trades = valuation_trades.div(self.portfolio_value, axis=0)
        return relative_trades.abs().sum(axis=1).mean() * self.periods_per_year

    @property
    def mean_return(self) -> float:
        return self.portfolio_returns.mean() * self.periods_per_year

    @property
    def volatility(self) -> float:
        return self.portfolio_returns.std() * np.sqrt(self.periods_per_year)

    @property
    def max_drawdown(self) -> float:
        return self.portfolio_value.div(self.portfolio_value.cummax()).sub(1).min()

    @property
    def max_leverage(self) -> float:
        return self.asset_weights.abs().sum(axis=1).max()

    @property
    def sharpe(self) -> float:
        risk_free = load_data()[3].loc[self.history]
        excess_return = self.portfolio_returns - risk_free
        return (
            excess_return.mean() / excess_return.std() * np.sqrt(self.periods_per_year)
        )

    def active_return(self, benchmark: BacktestResult) -> float:
        return self.mean_return - benchmark.mean_return

    def active_risk(self, benchmark: BacktestResult) -> float:
        return self.portfolio_returns.sub(benchmark.portfolio_returns).std() * np.sqrt(
            self.periods_per_year
        )

    def information_ratio(self, benchmark: BacktestResult) -> float:
        return self.active_return(benchmark) / self.active_risk(benchmark)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> BacktestResult:
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    # Example usage with equal weights
    n_assets = load_data()[0].shape[1]
    w_targets = np.ones(n_assets) / (n_assets + 1)
    c_target = 1 / (n_assets + 1)
    result = run_backtest(
        lambda _inputs: (w_targets, c_target, None), risk_target=0.0, verbose=True
    )
    print(
        f"Mean return: {result.mean_return:.2%},\n"
        f"Volatility: {result.volatility:.2%},\n"
        f"Sharpe: {result.sharpe:.2f},\n"
        f"Turnover: {result.turnover:.2f},\n"
        f"Max leverage: {result.max_leverage:.2f}"
    )
