import os

import cvxpy as cp
import numpy as np
import pandas as pd
from loguru import logger
from utils import generate_random_inputs, get_solver


def main(fitting: bool = False) -> None:
    scenarios = get_scenarios(fitting=fitting)
    res = []
    for n_assets, n_factors in scenarios:
        logger.info(f"Running scenario with {n_assets} assets and {n_factors} factors")
        n_iters = 1 if os.environ.get("CI") else 30
        for _ in range(n_iters):
            problem = run_scaling(n_assets, n_factors)
            assert problem.status in {
                cp.OPTIMAL,
                cp.OPTIMAL_INACCURATE,
            }, problem.status

            res.append(
                {
                    "n_assets": n_assets,
                    "n_factors": n_factors,
                    "solve_time": problem.solver_stats.solve_time,
                }
            )

    df = pd.DataFrame(res)

    df = df.groupby(["n_assets", "n_factors"]).mean().reset_index()

    if fitting:
        # Estimate the scaling exponents as solve time \approx a * n_assets^b * n_factors^c
        n_assets = df["n_assets"].values
        n_factors = df["n_factors"].values
        log_solve_time = np.log(df["solve_time"].values)

        a = cp.Variable()
        b = cp.Variable()
        c = cp.Variable()

        objective = cp.Minimize(
            cp.sum_squares(a + b * np.log(n_assets) + c * np.log(n_factors) - log_solve_time)
        )
        problem = cp.Problem(objective)
        problem.solve()
        assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}, problem.status

        logger.info(
            f"Estimated scaling exponents: a={np.exp(a.value):.2f}, \
            b={b.value:.2f}, c={c.value:.2f}"
        )

    else:
        df.set_index(["n_assets", "n_factors"], inplace=True)
        df = df.map(lambda x: f"{x:.2f}")

        # Reset column and row indices
        df.reset_index(inplace=True)
        df.columns.name = None
        df.index.name = None

        logger.info(df.to_latex(index=False))


def run_scaling(n_assets: int, n_factors: int) -> tuple[np.ndarray, float, cp.Problem]:
    mean, F, covariance = generate_random_inputs(n_assets, n_factors)
    factor_chol = np.linalg.cholesky(covariance)
    factor_volas = np.diag(factor_chol)

    equal_weights = np.ones(n_assets) / n_assets
    np.sqrt(equal_weights @ F @ covariance @ F.T @ equal_weights)
    sigma_target = 0

    # The risk constraint is soft.
    # For each percentage point of risk, we need to compensate with
    # 5 percentage points of return.

    rho_mean = np.percentile(np.abs(mean), 20, axis=0) * np.ones(n_assets)
    rho_covariance = 0.02
    L_max = 1.6
    T_max = 50 / 252 / 2

    risk_free = 0.0001
    w_lower = np.ones(n_assets) * (-0.05)
    w_upper = np.ones(n_assets) * 0.1
    c_lower = -0.05
    c_upper = 1.0
    gamma_risk = 5.0

    w_prev = np.ones(n_assets) / n_assets
    c_prev = 0.0

    w, c = cp.Variable(n_assets), cp.Variable()

    z = w - w_prev
    T = cp.norm1(z) / 2
    L = cp.norm1(w)

    # worst-case (robust) return
    mean_return = w @ mean + risk_free * c
    return_uncertainty = rho_mean @ cp.abs(w)
    return_wc = mean_return - return_uncertainty

    # worst-case (robust) risk
    risk = cp.norm2((F @ factor_chol).T @ w)
    factor_volas = cp.norm2(F @ factor_chol, axis=1)

    risk_uncertainty = rho_covariance**0.5 * factor_volas @ cp.abs(w)
    risk_wc = cp.norm2(cp.hstack([risk, risk_uncertainty]))

    objective = return_wc - gamma_risk * cp.pos(risk_wc - sigma_target)

    constraints = [
        cp.sum(w) + c == 1,
        c == c_prev - cp.sum(z),
        c_lower <= c,
        c <= c_upper,
        w_lower <= w,
        w <= w_upper,
        L <= L_max,
        T <= T_max,
    ]

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=get_solver(), verbose=False)

    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}, problem.status
    return problem


def get_scenarios(fitting: bool = False) -> list[tuple[int, int]]:
    if not fitting:
        return [
            (100, 10),
            (500, 20),
            (500, 50),
            (2_000, 50),
            (2_000, 100),
            (10_000, 50),
            (10_000, 100),
            (50_000, 200),
            (50_000, 500),
        ]
    else:
        # fine grid for fitting
        assets = np.logspace(3, 3.5, 10, dtype=int)
        factors = np.logspace(3, 3.5, 10, dtype=int)
        pairs = [(a, f) for a in assets for f in factors if a >= f]
        return pairs


if __name__ == "__main__":
    main()
