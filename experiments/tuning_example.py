import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import load_data
from .tuning_utils import (
    full_markowitz,
    leverages,
    risks,
    sharpes,
    tune_parameters,
    turnovers,
    yearly_data,
)
from .utils import experiment_path


def main() -> None:
    ### see if tuning_results is in results folder ###
    try:
        tuning_results = pd.read_csv(
            experiment_path() / "tuning_results/tuning_results.csv", index_col=0
        )
        gamma_holds = tuning_results.gamma_hold.to_list()
        gamma_trades = tuning_results.gamma_trade.to_list()
        gamma_turns = tuning_results.gamma_turn.to_list()
        gamma_leverages = tuning_results.gamma_leverage.to_list()
        gamma_risks = tuning_results.gamma_risk.to_list()
        backtests = tuning_results.index.to_list()
        sharpe_ratios_train = tuning_results.sharpe_train.to_list()
        sharpe_ratios_test = tuning_results.sharpe_test.to_list()
        volas_train = tuning_results.vola_train.to_list()
        volas_test = tuning_results.vola_test.to_list()
        leverages_train = tuning_results.lev_train.to_list()
        leverages_test = tuning_results.lev_test.to_list()
        turnovers_train = tuning_results.turn_train.to_list()
        turnovers_test = tuning_results.turn_test.to_list()

    except FileNotFoundError:
        prices, spread, rf, volume = load_data()
        all_prices, _, _, _ = yearly_data()

        prices_train = all_prices[15]
        time_last = prices_train.index[-1]

        prices_test = prices.loc[time_last:].iloc[1 : 1 + 250, :]

        prices_train_test = pd.concat([prices_train, prices_test], axis=0)
        spread_train_test = spread.loc[prices_train_test.index]
        volume_train_test = volume.loc[prices_train_test.index]
        rf_train_test = rf.loc[prices_train_test.index]

        parameter_dict, _ = tune_parameters(
            full_markowitz,
            prices_train_test,
            spread_train_test,
            volume_train_test,
            rf_train_test,
            train_len=500,
            verbose=True,
        )

        ### GET RESULTS ###
        sharpe_ratios_train = []
        sharpe_ratios_test = []
        volas_train = []
        volas_test = []
        leverages_train = []
        leverages_test = []
        turnovers_train = []
        turnovers_test = []

        gamma_holds = []
        gamma_trades = []
        gamma_turns = []
        gamma_leverages = []
        gamma_risks = []

        train_len = 500
        test_len = (
            len(prices_train_test) - 500 - train_len
        )  # first 500 use to initialize EWMA covariance predictions

        for i in parameter_dict.keys():
            sharpe_train, sharpe_test = sharpes(parameter_dict[i][-1], test_len)
            vola_train, vola_test = risks(parameter_dict[i][-1], test_len)
            leverage_train, leverage_test = leverages(parameter_dict[i][-1], test_len)
            turnover_train, turnover_test = turnovers(parameter_dict[i][-1], prices, test_len)

            sharpe_ratios_train.append(sharpe_train)
            sharpe_ratios_test.append(sharpe_test)

            volas_train.append(vola_train)
            volas_test.append(vola_test)

            leverages_train.append(leverage_train)
            leverages_test.append(leverage_test)

            turnovers_train.append(turnover_train)
            turnovers_test.append(turnover_test)

            gamma_holds.append(parameter_dict[i][0].gamma_hold)
            gamma_trades.append(parameter_dict[i][0].gamma_trade)
            gamma_turns.append(parameter_dict[i][0].gamma_turn)
            gamma_leverages.append(parameter_dict[i][0].gamma_leverage)
            gamma_risks.append(parameter_dict[i][0].gamma_risk)

        ### SAVE RESULTS ###
        backtests = np.arange(1, len(sharpe_ratios_train) + 1)
        df = pd.DataFrame(
            index=backtests,
            columns=[
                "gamma_hold",
                "gamma_trade",
                "gamma_turn",
                "gamma_leverage",
                "gamma_risk",
                "sharpe_train",
                "sharpe_test",
                "vola_train",
                "vola_test",
                "lev_train",
                "lev_test",
                "turn_train",
                "turn_test",
            ],
        )
        df.gamma_hold = gamma_holds
        df.gamma_trade = gamma_trades
        df.gamma_turn = gamma_turns
        df.gamma_leverage = gamma_leverages
        df.gamma_risk = gamma_risks
        df.sharpe_train = sharpe_ratios_train
        df.sharpe_test = sharpe_ratios_test
        df.vola_train = volas_train
        df.vola_test = volas_test
        df.lev_train = leverages_train
        df.lev_test = leverages_test
        df.turn_train = turnovers_train
        df.turn_test = turnovers_test
        df.to_csv(experiment_path() / "tuning_results/tuning_results.csv")

    ### PLOT RESULTS ###
    plot_results(
        backtests,
        sharpe_ratios_train,
        sharpe_ratios_test,
        volas_train,
        volas_test,
        leverages_train,
        leverages_test,
        turnovers_train,
        turnovers_test,
    )


def plot_results(
    backtests: np.ndarray,
    sharpe_ratios_train: np.ndarray,
    sharpe_ratios_test: np.ndarray,
    volas_train: np.ndarray,
    volas_test: np.ndarray,
    leverages_train: np.ndarray,
    leverages_test: np.ndarray,
    turnovers_train: np.ndarray,
    turnovers_test: np.ndarray,
) -> None:
    show_plot = False

    # sharpe
    plt.plot(backtests, sharpe_ratios_train, label="in-sample", marker="o")
    plt.plot(backtests, sharpe_ratios_test, label="out-of-sample", marker="o")
    plt.ylabel("Sharpe ratio")
    plt.xlabel("Number of backtests")
    plt.xticks(backtests)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))
    plt.legend()
    plt.ylim(3, 7)
    plt.savefig(experiment_path() / "tuning_results/tuning_SR.pdf", bbox_inches="tight")
    if show_plot:
        plt.show()

    # vola
    plt.plot(backtests, volas_train, label="train", marker="o")
    plt.plot(backtests, volas_test, label="test", marker="o")
    plt.ylabel("Volatility")
    plt.xlabel("Number of backtests")
    plt.xticks(backtests)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))
    plt.ylim(0.0, 0.12)
    plt.savefig(experiment_path() / "tuning_results/tuning_vola.pdf", bbox_inches="tight")
    if show_plot:
        plt.show()

    # leverage
    plt.plot(backtests, leverages_train, label="train", marker="o")
    plt.plot(backtests, leverages_test, label="test", marker="o")
    plt.ylabel("Leverage")
    plt.xlabel("Number of backtests")
    plt.xticks(backtests)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))
    plt.ylim(1, 2)
    plt.savefig(experiment_path() / "tuning_results/tuning_lev.pdf", bbox_inches="tight")
    if show_plot:
        plt.show()

    # turnover
    plt.plot(backtests, turnovers_train, label="train", marker="o")
    plt.plot(backtests, turnovers_test, label="test", marker="o")
    plt.ylabel("Turnover")
    plt.xlabel("Number of backtests")
    plt.xticks(backtests)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))
    plt.ylim(20, 50)
    plt.savefig(experiment_path() / "tuning_results/tuning_turn.pdf", bbox_inches="tight")
    if show_plot:
        plt.show()


if __name__ == "__main__":
    main()
