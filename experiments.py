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
