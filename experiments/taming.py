# PLEASE DO NOT USE Notebook files in here...
from pathlib import Path

import numpy as np

from markowitz import Data


def data_folder():
    return Path(__file__).parent.parent / "data"


def main():
    print("Hello World")
    print(data_folder())
    data = Data
    data.idio_volas = np.array([1, 2, 3])
    print(data)


if __name__ == "__main__":
    main()
