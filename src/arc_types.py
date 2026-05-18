from dataclasses import dataclass
from typing import TypeAlias
import numpy as np

# A grid is a 2-D numpy array of ints 0-9 (ARC color indices)
Grid: TypeAlias = np.ndarray

# ARC color palette (index → name)
COLORS = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "fuchsia",
    7: "orange",
    8: "azure",
    9: "maroon",
}

# Matplotlib-compatible hex colors matching the ARC UI
COLOR_HEX = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 grey
    "#F012BE",  # 6 fuchsia
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 azure
    "#870C25",  # 9 maroon
]


@dataclass
class Example:
    input: Grid
    output: Grid


@dataclass
class Task:
    task_id: str
    train: list[Example]
    test: list[Example]  # output may be None for final test tasks

    @property
    def n_train(self) -> int:
        return len(self.train)
