from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .arc_types import COLOR_HEX


class Grid:
    """An ARC grid storing color values 0-9 as a 2-D numpy array."""

    def __init__(self, data: list[list[int]] | np.ndarray) -> None:
        if isinstance(data, np.ndarray):
            self._data = data.astype(np.int8)
        else:
            self._data = np.array(data, dtype=np.int8)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def height(self) -> int:
        return self._data.shape[0]

    @property
    def width(self) -> int:
        return self._data.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    def show(self, ax: plt.Axes | None = None, title: str = "") -> None:
        """Render the grid with ARC colors. Pass *ax* to embed in an existing figure."""
        cmap = mcolors.ListedColormap(COLOR_HEX)
        norm = mcolors.BoundaryNorm(boundaries=range(11), ncolors=10)

        standalone = ax is None
        if standalone:
            scale = 0.5
            fig, ax = plt.subplots(
                figsize=(max(self.width * scale, 2), max(self.height * scale, 2))
            )

        ax.imshow(self._data, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        if title:
            ax.set_title(title)

        if standalone:
            plt.tight_layout()
            plt.show()

    def __repr__(self) -> str:
        return f"Grid(shape={self.shape})"
