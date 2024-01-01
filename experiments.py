import os
from pathlib import Path

from experiments.taming import main as taming_main
from experiments.scaling_small import main as scaling_small_main
from experiments.scaling_large import main as scaling_large_main

if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    scaling_small_main()

    if not os.getenv("CI"):
        scaling_large_main()
        taming_main()
