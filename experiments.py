from pathlib import Path

from experiments.taming import main as taming_main
from experiments.scaling_small import main as scaling_small_main

if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    scaling_small_main()
    taming_main()
