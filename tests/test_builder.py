# """
# testing the builder
# """

# import sys
# from pathlib import Path
#
# sys.path.append(Path(__file__).parent.parent / "cvx")

from cvx.markowitz.backtest import ParameterizedProblem


def test_trivial(prices):
    assert prices.shape == (602, 7)


def test_problem():
    p = ParameterizedProblem(
        problem="Markowitz",
        variables=["x", "y"],
        parameters="x + y",
    )

    assert p
