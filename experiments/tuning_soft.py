import numpy as np
from loguru import logger

from experiments.tuning_utils import (
    HyperParameters,
    Targets,
    full_markowitz,
    run_soft_backtest,
)

targets = Targets(
    T_max=25 / 252,
    L_max=1.6,
    risk_target=0.1 / np.sqrt(252),
)


# Initialize hyperparameters from tuning_duals.py
hyperparameters = HyperParameters(1, 1, 5e-3, 5e-4, 5e-2)

if __name__ == "__main__":
    results_soft = run_soft_backtest(
        full_markowitz,
        targets,
        hyperparameters,
        None,
        None,
        None,
        None,
        verbose=True,
    )

    logger.info(f"mean: {results_soft.mean_return:.1%}")
    logger.info(f"volatility: {results_soft.volatility:.1%}")
    logger.info(f"max drawdown: {results_soft.max_drawdown:.1%}")
    logger.info(f"max leverage: {results_soft.max_leverage:.2f}")
    logger.info(f"sharpe: {results_soft.sharpe:.2f}")
    logger.info(f"turnover: {results_soft.turnover:.2f}")
