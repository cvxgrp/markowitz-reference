from pathlib import Path

from experiments.taming import main as taming_main

if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    taming_main()
