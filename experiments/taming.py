# PLEASE DO NOT USE Notebook files in here...

from markowitz import foo
from pathlib import Path


def data_folder():
    return Path(__file__).parent.parent / "data"


if __name__ == "__main__":
    print("Hello World")
    print(foo())
    print(data_folder())
