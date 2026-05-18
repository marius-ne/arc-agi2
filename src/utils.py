import json
import os
from pathlib import Path
import numpy as np
from .arc_types import Grid, Task, Example


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_task(path: str | Path) -> Task:
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)
    task_id = path.stem
    train = [Example(np.array(e["input"]), np.array(e["output"])) for e in raw["train"]]
    test = []
    for e in raw["test"]:
        inp = np.array(e["input"])
        out = np.array(e["output"]) if "output" in e else None
        test.append(Example(inp, out))
    return Task(task_id=task_id, train=train, test=test)


def load_tasks_from_dir(directory: str | Path) -> list[Task]:
    directory = Path(directory)
    tasks = []
    for f in sorted(directory.glob("*.json")):
        tasks.append(load_task(f))
    return tasks


def save_submission(predictions: dict[str, list[Grid]], path: str | Path) -> None:
    """Save predictions in ARC submission format: {task_id: [[attempt1, attempt2], ...]}"""
    path = Path(path)
    output = {}
    for task_id, grids in predictions.items():
        output[task_id] = [g.tolist() for g in grids]
    with open(path, "w") as f:
        json.dump(output, f)


# ---------------------------------------------------------------------------
# Grid constructors
# ---------------------------------------------------------------------------

def grid_from_list(data: list[list[int]]) -> Grid:
    return np.array(data, dtype=np.int8)


def empty_grid(h: int, w: int, fill: int = 0) -> Grid:
    return np.full((h, w), fill, dtype=np.int8)


# ---------------------------------------------------------------------------
# Geometric transforms
# ---------------------------------------------------------------------------

def rotate90(grid: Grid, k: int = 1) -> Grid:
    return np.rot90(grid, k=k)


def flip_h(grid: Grid) -> Grid:
    return np.fliplr(grid)


def flip_v(grid: Grid) -> Grid:
    return np.flipud(grid)


def transpose(grid: Grid) -> Grid:
    return grid.T.copy()


# ---------------------------------------------------------------------------
# Cropping / padding
# ---------------------------------------------------------------------------

def bounding_box(grid: Grid, background: int = 0) -> tuple[int, int, int, int]:
    """Return (row_min, row_max, col_min, col_max) of non-background cells."""
    rows = np.any(grid != background, axis=1)
    cols = np.any(grid != background, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return int(r_min), int(r_max), int(c_min), int(c_max)


def crop(grid: Grid, background: int = 0) -> Grid:
    r0, r1, c0, c1 = bounding_box(grid, background)
    return grid[r0:r1+1, c0:c1+1].copy()


def pad(grid: Grid, top: int = 0, bottom: int = 0, left: int = 0, right: int = 0,
        fill: int = 0) -> Grid:
    return np.pad(grid, ((top, bottom), (left, right)), constant_values=fill)


# ---------------------------------------------------------------------------
# Color / cell queries
# ---------------------------------------------------------------------------

def unique_colors(grid: Grid) -> list[int]:
    return sorted(np.unique(grid).tolist())


def color_mask(grid: Grid, color: int) -> np.ndarray:
    return grid == color


def replace_color(grid: Grid, src: int, dst: int) -> Grid:
    g = grid.copy()
    g[g == src] = dst
    return g


def color_counts(grid: Grid) -> dict[int, int]:
    values, counts = np.unique(grid, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------

def connected_components(grid: Grid, color: int | None = None,
                          connectivity: int = 4) -> list[list[tuple[int, int]]]:
    """
    Return a list of components; each component is a list of (row, col) cells.
    If color is None, treat all non-zero cells together.
    connectivity: 4 or 8.
    """
    from collections import deque
    mask = (grid != 0) if color is None else (grid == color)
    visited = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dirs8 = dirs4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    dirs = dirs8 if connectivity == 8 else dirs4

    components = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not visited[r, c]:
                component = []
                q = deque([(r, c)])
                visited[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    component.append((cr, cc))
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                components.append(component)
    return components
