from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import pickle
import sys
from typing import Callable
import numpy as np
import pandas as pd
from utils import synthetic_returns

# hack to allow importing from parent directory without having a package
sys.path.append(str(Path(__file__).parent.parent))


def data_folder():
    return Path(__file__).parent.parent / "data"


@lru_cache(maxsize=1)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(data_folder() / "prices.csv", index_col=0, parse_dates=True)
    spread = pd.read_csv(data_folder() / "spreads.csv", index_col=0, parse_dates=True)
    volume = pd.read_csv(data_folder() / "volumes.csv", index_col=0, parse_dates=True)
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


def run_backtest(
    strategy: Callable, risk_target: float, verbose: bool = False
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Run a simplified backtest for a given strategy.
    At time t we use data from t-500 to t-1 to forecast the data and
    compute the optimal portfolio weights and cash holdings.
    We then trade to these weights at time t.
    """

    prices, spread, volume, rf = load_data()
    n_assets = prices.shape[1]

    lookback = 500

    quantities = np.zeros(n_assets)
    cash = 1e6

    post_trade_cash = []
    post_trade_quantities = []

    returns = synthetic_returns(prices).dropna()
    means = returns.ewm(halflife=125).mean()
    covariance_df = returns.ewm(halflife=125).cov()
    days = returns.index
    covariances = {}
    for day in days:
        covariances[day] = covariance_df.loc[day]

    for t in range(lookback, len(prices)):
        day = prices.index[t]

        if verbose:
            print(f"Day {t} of {len(prices)-1}, {day}")

        prices_t = prices.iloc[t - lookback : t]  # Up to t-1
        spread_t = spread.iloc[t - lookback : t]
        volume_t = volume.iloc[t - lookback : t]
        mean_t = means.loc[day]
        covariance_t = covariances[day]

        inputs_t = OptimizationInput(
            prices_t,
            mean_t,
            covariance_t,
            spread_t,
            volume_t,
            quantities,
            cash,
            risk_target,
        )

        w, _ = strategy(inputs_t)

        latest_prices = prices.iloc[t]  # At t
        latest_spread = spread.iloc[t]

        cash += interest_and_fees(
            cash, rf.iloc[t - 1], quantities, prices.iloc[t - 1], day
        )
        trade_quantities = create_orders(w, quantities, cash, latest_prices)
        quantities += trade_quantities
        cash += execute_orders(latest_prices, trade_quantities, latest_spread)

        post_trade_cash.append(cash)
        post_trade_quantities.append(quantities)

    post_trade_cash = pd.Series(post_trade_cash, index=prices.index[lookback:])
    post_trade_quantities = pd.DataFrame(
        post_trade_quantities, index=prices.index[lookback:], columns=prices.columns
    )
    return BacktestResult(post_trade_cash, post_trade_quantities, risk_target)


def create_orders(w, quantities, cash, latest_prices) -> np.array:
    portfolio_value = cash + quantities @ latest_prices
    w_prev = (quantities * latest_prices) / portfolio_value

    z = w - w_prev
    trades = z * portfolio_value
    trade_quantities = trades / latest_prices
    return trade_quantities.values


def execute_orders(latest_prices, trade_quantities, latest_spread) -> float:
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
    shorting_fee = short_value * (1 + rf) ** days_t_to_t_minus_1 - short_value
    return cash_interest + shorting_fee


@dataclass
class BacktestResult:
    cash: pd.Series
    quantities: pd.DataFrame
    risk_target: float

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
        return (
            self.asset_weights.diff().abs().sum(axis=1).mean() * self.periods_per_year
        )

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
        return self.mean_return / self.volatility  # TODO: risk free rate

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "BacktestResult":
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    # Example usage with equal weights
    n_assets = load_data()[0].shape[1]
    w_targets = np.ones(n_assets) / (n_assets + 1)
    c_target = 1 / (n_assets + 1)
    run_backtest(lambda _inputs: (w_targets, c_target), risk_target=0.0, verbose=True)
