from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import pickle
import sys
import numpy as np
import cvxpy as cp
import pandas as pd

# hack to allow importing from parent directory without having a package
sys.path.append(str(Path(__file__).parent.parent))


def data_folder():
    return Path(__file__).parent.parent / "data"


@lru_cache(maxsize=1)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(data_folder() / "prices.csv", index_col=0, parse_dates=True)
    spread = pd.read_csv(data_folder() / "spreads.csv", index_col=0, parse_dates=True)
    volume = (
        pd.read_csv(data_folder() / "volumes_shares.csv", index_col=0, parse_dates=True)
        * prices
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
    # risk_target: float
    risk_free: float

    @property
    def n_assets(self) -> int:
        return self.prices.shape[1]


def run_backtest():
    # TODO: add back Philipps code
    pass


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
) -> float:
    sell_order_quantities = np.clip(trade_quantities, None, 0)
    buy_order_quantities = np.clip(trade_quantities, 0, None)

    sell_order_prices = latest_prices * (1 - latest_spread / 2)
    buy_order_prices = latest_prices * (1 + latest_spread / 2)

    sell_receipt = -sell_order_quantities @ sell_order_prices
    buy_payment = buy_order_quantities @ buy_order_prices

    return sell_receipt - buy_payment


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
    short_spread = 0.05 / 360
    shorting_fee = (
        short_value * (1 + rf + short_spread) ** days_t_to_t_minus_1 - short_value
    )
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
    dual_optimals: pd.DataFrame = field(default_factory=pd.DataFrame)

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
