import numpy as np
from loguru import logger

from experiments.backtest import load_data
from experiments.tuning_utils import (
    HyperParameters,
    Targets,
    run_hard_backtest,
    solve_hard_markowitz,
)


def main() -> None:
    prices, spread, rf, volume = load_data()
    training_length = 1250
    burn_in = 500  # first 500 for covariance

    training_length = training_length + burn_in
    prices, spread, rf, volume = (
        prices.iloc[:training_length],
        spread.iloc[:training_length],
        rf.iloc[:training_length],
        volume.iloc[:training_length],
    )

    targets = Targets(
        T_max=25 / 252,
        L_max=1.6,
        risk_target=0.1 / np.sqrt(252),
    )

    hyperparameters = HyperParameters(1, 1, 0, 0, 0)

    results = run_hard_backtest(
        solve_hard_markowitz,
        targets,
        hyperparameters,
        prices,
        spread,
        volume,
        rf,
        verbose=True,
    )

    gamma_risk = results.dual_optimals.Risk.quantile(0.7)
    gamma_turn = results.dual_optimals.Turnover.quantile(0.7)
    gamma_lev = 0.25 * results.dual_optimals.Leverage.quantile(1)

    logger.info(f"\ngamma_risk: {gamma_risk}")
    logger.info(f"gamma_turn: {gamma_turn}")
    logger.info(f"gamma_lev: {gamma_lev}")


if __name__ == "__main__":
    main()
