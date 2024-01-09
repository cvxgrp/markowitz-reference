from functools import lru_cache
import time

from loguru import logger
from matplotlib import pyplot as plt
import cvxpy as cp
import numpy as np

from experiments.backtest import (
    BacktestResult,
    OptimizationInput,
    Timing,
    load_data,
    run_backtest,
)
from experiments.utils import get_solver, checkpoints_path, figures_path


def parameter_scaling_markowitz(
    inputs: OptimizationInput,
) -> tuple[np.ndarray, float, cp.Problem]:
    problem, param_dict, w, c = get_parametrized_problem(
        inputs.n_assets, inputs.risk_target
    )
    latest_prices = inputs.prices.iloc[-1]
    portfolio_value = inputs.cash + inputs.quantities @ latest_prices

    param_dict["chol"].value = inputs.chol
    param_dict["volas"].value = inputs.volas
    param_dict["rho_mean"].value = np.percentile(
        np.abs(inputs.mean.values), 20, axis=0
    ) * np.ones(inputs.n_assets)
    param_dict["w_prev"].value = (
        inputs.quantities * inputs.prices.iloc[-1] / portfolio_value
    ).values
    param_dict["c_prev"].value = inputs.cash / portfolio_value
    param_dict["mean"].value = inputs.mean.values
    param_dict["risk_free"].value = inputs.risk_free

    problem.solve(solver=get_solver())
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}, problem.status
    return w.value, c.value, problem


@lru_cache
def get_parametrized_problem(
    n_assets: int, risk_target: float
) -> tuple[cp.Problem, dict, cp.Variable, cp.Variable]:
    rho_covariance = 0.02
    L_max = 1.6
    T_max = 50 / 252 / 2

    w_lower = np.ones(n_assets) * (-0.05)
    w_upper = np.ones(n_assets) * 0.1
    c_lower = -0.05
    c_upper = 1.0
    gamma_risk = 5.0

    w_prev = cp.Parameter(n_assets)
    c_prev = cp.Parameter()
    mean = cp.Parameter(n_assets)
    risk_free = cp.Parameter()
    rho_mean = cp.Parameter(n_assets)
    chol = cp.Parameter((n_assets, n_assets))
    volas = cp.Parameter(n_assets, nonneg=True)

    w, c = cp.Variable(n_assets), cp.Variable()

    z = w - w_prev
    T = cp.norm1(z) / 2
    L = cp.norm1(w)

    # worst-case (robust) return
    mean_return = w @ mean + risk_free * c
    abs_weight_var = cp.Variable(n_assets, nonneg=True)
    return_uncertainty = rho_mean @ abs_weight_var
    return_wc = mean_return - return_uncertainty

    # worst-case (robust) risk
    risk = cp.norm2(chol.T @ w)
    risk_uncertainty = rho_covariance**0.5 * volas @ abs_weight_var
    risk_wc = cp.norm2(cp.hstack([risk, risk_uncertainty]))

    objective = return_wc - gamma_risk * cp.pos(risk_wc - risk_target)

    constraints = [
        cp.sum(w) + c == 1,
        c == c_prev - cp.sum(z),
        c_lower <= c,
        c <= c_upper,
        w_lower <= w,
        w <= w_upper,
        L <= L_max,
        T <= T_max,
        cp.abs(w) <= abs_weight_var,
    ]

    problem = cp.Problem(cp.Maximize(objective), constraints)

    param_dict = {
        "w_prev": w_prev,
        "mean": mean,
        "risk_free": risk_free,
        "rho_mean": rho_mean,
        "chol": chol,
        "volas": volas,
        "c_prev": c_prev,
    }
    return problem, param_dict, w, c


def plot_timings(timings: list[Timing]) -> None:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    line_color = "#888888"
    plt.figure()
    plt.stackplot(
        range(len(timings)),
        [timing.cvxpy for timing in timings],
        [timing.solver for timing in timings],
        [timing.other for timing in timings],
        labels=["CVXPY", "Solver", "Other"],
        colors=colors,
    )

    # add light horizontal line for average solver time
    average_cvxpy_time = np.mean([timing.cvxpy for timing in timings])
    average_solver_time = np.mean([timing.solver for timing in timings])
    average_other_time = np.mean([timing.other for timing in timings])

    plt.axhline(
        average_cvxpy_time,
        color=line_color,
        linestyle="--",
    )

    plt.axhline(
        average_solver_time + average_cvxpy_time,
        color=line_color,
        linestyle="--",
    )

    plt.axhline(
        average_other_time + average_solver_time + average_cvxpy_time,
        color=line_color,
        linestyle="--",
    )

    plt.xlabel("Day of backtest")
    plt.xlim(0, len(timings))

    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig(figures_path() / "timing_parametrized.pdf")
    plt.show()


def initialize_problem(n_assets: int, sigma_target: float) -> None:
    start = time.perf_counter()
    problem, param_dict, _, _ = get_parametrized_problem(n_assets, sigma_target)

    try:
        for p in param_dict.values():
            p.value = np.zeros(p.shape)
        problem.solve(solver=get_solver())
    except cp.SolverError:
        pass

    end = time.perf_counter()
    logger.info(f"First call to get_parametrized_problem took {end-start} seconds")


def main(from_checkpoint: bool = False) -> None:
    annualized_target = 0.10
    sigma_target = annualized_target / np.sqrt(252)

    if not from_checkpoint:
        logger.info("Running parameter scaling")

        n_assets = load_data()[0].shape[1]

        initialize_problem(n_assets, sigma_target)

        scaling_parametrized_markowitz_result = run_backtest(
            parameter_scaling_markowitz, sigma_target, verbose=True
        )
        scaling_parametrized_markowitz_result.save(
            checkpoints_path() / f"scaling_parametrized_{annualized_target}.pickle"
        )
    else:
        scaling_parametrized_markowitz_result = BacktestResult.load(
            checkpoints_path() / f"scaling_parametrized_{annualized_target}.pickle"
        )

    total_time = sum(t.total for t in scaling_parametrized_markowitz_result.timings)
    cvxpy_time = sum(t.cvxpy for t in scaling_parametrized_markowitz_result.timings)
    solver_time = sum(t.solver for t in scaling_parametrized_markowitz_result.timings)
    other_time = sum(t.other for t in scaling_parametrized_markowitz_result.timings)
    logger.info(f"Total time {total_time}")
    logger.info(len(scaling_parametrized_markowitz_result.timings))
    logger.info(
        f"Average time {total_time / len(scaling_parametrized_markowitz_result.timings)}"
    )
    logger.info(
        f"Average solver time {solver_time / len(scaling_parametrized_markowitz_result.timings)}"
    )
    logger.info(f"CVXPY time {cvxpy_time/total_time}")
    logger.info(f"Solver time {solver_time/total_time}")
    logger.info(f"Other time {other_time/total_time}")
    plot_timings(scaling_parametrized_markowitz_result.timings)


if __name__ == "__main__":
    main()
