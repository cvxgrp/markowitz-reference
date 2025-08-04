"""
This script runs the experiments for the Markowitz reference implementation.

To run this script with uv (https://github.com/astral-sh/uv):
1. Install uv:
   curl -LsSf https://astral.sh/uv/install.sh | sh

2. Run the experiments:
   uv run experiments.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mosek==11.0.27",
#   "loguru==0.7.3",
#   "numpy==2.3.2",
#   "pandas[output-formatting]==2.3.1",
#   "matplotlib==3.10.5",
#   "cvxpy-base==1.7.1",
#   "clarabel==0.11.1"
# ]
# ///
import os
import sys
from pathlib import Path

from loguru import logger

sys.path.append((Path(__file__).parent / "experiments").as_posix())

from experiments.scaling_large import main as scaling_large_main
from experiments.scaling_small import main as scaling_small_main
from experiments.taming import main as taming_main
from experiments.utils import checkpoints_path, figures_path

if __name__ == "__main__":
    logger.info("sys.path:")
    for path in sys.path:
        logger.debug(path)

    logger.debug("Create paths for checkpoints and figures")
    checkpoints_path().mkdir(exist_ok=True)
    figures_path().mkdir(exist_ok=True)

    logger.debug("Run experiments")
    scaling_small_main()

    if not os.getenv("CI"):
        # Large scale experiments require a Mosek license
        # Hence we do not perform them on a GitHub CI server
        scaling_large_main(fitting=True)
        scaling_large_main(fitting=False)
        taming_main()
