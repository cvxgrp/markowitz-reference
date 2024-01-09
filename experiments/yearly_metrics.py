import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tuning_utils import (
    HyperParameters,
    Targets,
    full_markowitz,
    run_soft_backtest,
)
from utils import experiment_path

if __name__ == "__main__":
    try:
        parameters_df = pd.read_csv(
            experiment_path() / "tuning_results/parameters.csv",
            index_col=0,
            parse_dates=True,
        )
        parameters_df.index = pd.to_datetime(parameters_df.index)
    except FileNotFoundError as exc:
        raise FileNotFoundError("Run 'tuning_yearly_retune.py' first.") from exc

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

    ### Yearly metrics ###
    port_returns = results_fully_tuned.portfolio_returns

    yearly_means_tuned = port_returns.resample("Y").mean() * results_fully_tuned.periods_per_year
    yearly_volas_tuned = port_returns.resample("Y").std() * np.sqrt(
        results_fully_tuned.periods_per_year
    )
    yearly_sharpes_tuned = yearly_means_tuned / yearly_volas_tuned

    ### Benchmarks from the 'taming.py' experiments ###
    equal_means = (
        pd.Series(
            np.array(
                [
                    33.6,
                    13.1,
                    -26.3,
                    38.6,
                    17.6,
                    6.2,
                    20.3,
                    30.7,
                    15.8,
                    4.3,
                    15.0,
                    20.9,
                    -1.9,
                    27.8,
                    20.1,
                    27.3,
                    -5.9,
                    10.7,
                ]
            ),
            index=yearly_means_tuned.index,
        )
        / 100
    )

    basic_means = (
        pd.Series(
            [
                -52.6,
                -69.7,
                -13.5,
                6.7,
                -45.6,
                0.4,
                21.8,
                -0.0,
                -16.3,
                11.2,
                9.9,
                23.5,
                22.1,
                2.1,
                70.6,
                5.6,
                34.2,
                25.0,
            ],
            index=yearly_means_tuned.index,
        )
        / 100
    )

    equal_volas = (
        pd.Series(
            [
                8.5,
                15.6,
                42.0,
                30.1,
                17.3,
                22.8,
                12.3,
                11.0,
                10.8,
                15.1,
                12.8,
                6.6,
                16.0,
                12.0,
                35.8,
                12.2,
                20.8,
                12.4,
            ],
            index=yearly_means_tuned.index,
        )
        / 100
    )
    basic_volas = (
        pd.Series(
            [
                13.5,
                15.3,
                23.5,
                13.8,
                9.7,
                11.0,
                12.5,
                11.8,
                10.8,
                13.4,
                14.4,
                13.9,
                15.2,
                13.2,
                21.6,
                11.2,
                14.0,
                12.3,
            ],
            index=yearly_means_tuned.index,
        )
        / 100
    )

    equal_sharpes = pd.Series(
        [
            3.51,
            0.61,
            -0.66,
            1.28,
            1.01,
            0.27,
            1.64,
            2.78,
            1.46,
            0.28,
            1.15,
            3.08,
            -0.20,
            2.17,
            0.55,
            2.24,
            -0.34,
            0.58,
        ],
        index=yearly_means_tuned.index,
    )
    basic_sharpes = pd.Series(
        [
            -4.18,
            -4.81,
            -0.64,
            0.47,
            -4.71,
            0.03,
            1.74,
            -0.01,
            -1.51,
            0.82,
            0.67,
            1.64,
            1.37,
            0.04,
            3.25,
            0.49,
            2.35,
            1.74,
        ],
        index=yearly_means_tuned.index,
    )

    # means
    yearly_means_tuned.plot(label="Tuned Markowitz++", marker="o")
    equal_means.plot(label="Equal weights", marker="o")
    basic_means.plot(label="Basic Markowitz", marker="o")
    plt.ylabel("Mean return")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x * 100):,}%"))
    plt.legend()
    plt.savefig(experiment_path() / "tuning_results/yearly_means.pdf", bbox_inches="tight")
    plt.show()

    # volas
    yearly_volas_tuned.plot(label="Tuned Markowitz++", marker="o")
    equal_volas.plot(label="Equal weights", marker="o")
    basic_volas.plot(label="Basic Markowitz", marker="o")
    plt.ylabel("Volatility")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x * 100):,}%"))
    plt.savefig(experiment_path() / "tuning_results/yearly_volas.pdf", bbox_inches="tight")
    plt.show()

    # sharpes
    yearly_sharpes_tuned.plot(label="Tuned Markowitz++", marker="o")
    equal_sharpes.plot(label="Equal weights", marker="o")
    basic_sharpes.plot(label="Basic Markowitz", marker="o")
    plt.ylabel("Sharpe ratio")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))
    plt.legend()
    plt.savefig(experiment_path() / "tuning_results/yearly_sharpes.pdf", bbox_inches="tight")
    plt.show()
