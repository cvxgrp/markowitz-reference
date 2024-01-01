from pathlib import Path

from loguru import logger

from experiments.taming import main as taming_main
from experiments.scaling_small import main as scaling_small_main
from experiments.scaling_large import main as scaling_large_main

if __name__ == "__main__":
    logger.info("Welcome to ...")

    logger.debug("Create paths for checkpoints and figures")
    Path("checkpoints").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    logger.debug("Run experiments")
    scaling_small_main(logger=logger)
    scaling_large_main(logger=logger)
    taming_main(logger=logger)
