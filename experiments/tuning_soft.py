import numpy as np

from experiments.tuning_utils import (
    run_soft_backtest,
    HyperParameters,
    Targets,
    full_markowitz,
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

    print(f"mean: {results_soft.mean_return:.1%}")
    print(f"volatility: {results_soft.volatility:.1%}")
    print(f"max drawdown: {results_soft.max_drawdown:.1%}")
    print(f"max leverage: {results_soft.max_leverage:.2f}")
    print(f"sharpe: {results_soft.sharpe:.2f}")
    print(f"turnover: {results_soft.turnover:.2f}")
