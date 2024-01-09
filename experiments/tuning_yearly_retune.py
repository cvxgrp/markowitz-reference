import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import experiment_path
from backtest import load_data
from tuning_utils import (
    tune_in_parallel,
    HyperParameters,
    Targets,
    run_soft_backtest,
    full_markowitz,
    yearly_data,
)


if __name__ == "__main__":
    # SEE IF tuning_results/parameters.csv exists
    # try to load it

    try:
        parameters_df = pd.read_csv(
            experiment_path() / "tuning_results/parameters.csv",
            index_col=0,
            parse_dates=True,
        )
        parameters_df.index = pd.to_datetime(parameters_df.index)
    except FileNotFoundError:
        prices, spread, rf, volume = load_data()
        all_prices, all_spreads, all_volumes, all_rfs = yearly_data()

        tuning_results = tune_in_parallel(
            all_prices,
            all_spreads,
            all_volumes,
            all_rfs,
        )

        ### RUN WITH TUNED PARAMETERS ###
        parameters = {}
        for i, res in enumerate(tuning_results):
            time = all_prices[i].index[-1]
            best_backtest = res[-1]
            parameters[time] = res[0][best_backtest][0]

        parameters_df = pd.DataFrame(
            index=prices.index,
            columns=[
                "gamma_hold",
                "gamma_trade",
                "gamma_turn",
                "gamma_leverage",
                "gamma_risk",
            ],
        )

        for time in parameters.keys():
            params = parameters[time]
            parameters_df.loc[time] = [
                params.gamma_hold,
                params.gamma_trade,
                params.gamma_turn,
                params.gamma_leverage,
                params.gamma_risk,
            ]

        default = HyperParameters(1, 1, 5e-3, 5e-4, 5e-2)

        parameters_df.iloc[0] = [
            default.gamma_hold,
            default.gamma_trade,
            default.gamma_turn,
            default.gamma_leverage,
            default.gamma_risk,
        ]

        parameters_df = parameters_df.ffill()

        ### Save parameters to csv ###
        parameters_df.to_csv(experiment_path() / "tuning_results/parameters.csv")

    hyperparameters = HyperParameters(
        gamma_hold=parameters_df.gamma_hold,
        gamma_trade=parameters_df.gamma_trade,
        gamma_turn=parameters_df.gamma_turn,
        gamma_leverage=parameters_df.gamma_leverage,
        gamma_risk=parameters_df.gamma_risk,
    )

    targets = Targets(
        T_max=25 / 252,
        L_max=1.6,
        risk_target=0.1 / np.sqrt(252),
    )

    results_fully_tuned = run_soft_backtest(
        full_markowitz,
        targets=targets,
        hyperparameters=hyperparameters,
        verbose=True,
    )

    ### PRINT RESULTS ###
    print(f"mean: {results_fully_tuned.mean_return:.1%}")
    print(f"volatility: {results_fully_tuned.volatility:.1%}")
    print(f"max drawdown: {results_fully_tuned.max_drawdown:.1%}")
    print(f"max leverage: {results_fully_tuned.max_leverage:.2f}")
    print(f"sharpe: {results_fully_tuned.sharpe:.2f}")
    print(f"turnover: {results_fully_tuned.turnover:.2f}")

    ### PLOT RESULTS ###
    tuning_len = 1250
    for param in parameters_df.columns:
        plt.figure()
        plt.plot(parameters_df[param].iloc[tuning_len:], label=param)
        # fix y axis text to not be too wide and write as x 10^x
        plt.gca().get_yaxis().get_major_formatter().set_powerlimits((0, 1))
        plt.savefig(
            experiment_path() / f"tuning_results/tuning_{param}.pdf",
            bbox_inches="tight",
        )
        plt.show()

        # save figure as pdf
