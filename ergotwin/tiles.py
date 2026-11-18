"""Recordings that show several camera views in a grid: analyse one view, or every view at once.

The people in a grid are small: in a 3 x 3 grid each view has a third of the width. So the app
cuts each view out and enlarges it before looking for people, and each view is treated as a
camera of its own, with its own perspective and its own tracks.
"""

from __future__ import annotations

import numpy as np

WHOLE = "whole"
ALL = "all"
MAX_SIDE = 6  # grids up to 6 x 6

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels


def parse_tile(spec: str | None) -> tuple[int, int, int | None] | None:
    """Which part of the picture to analyse.

    'whole' or empty -> None: the whole picture, as it is.
    '3x3:5' -> (3 columns, 3 rows, 5): tile 5 alone, enlarged.
    '3x3:all' -> (3, 3, None): every tile, each enlarged and analysed on its own.

    Tiles are counted from 1, left to right, then top to bottom.
    """
    text = (spec or "").strip().lower().replace("×", "x")
    if text in ("", WHOLE):
        return None
    try:
        grid, _, index = text.partition(":")
        cols, _, rows = grid.partition("x")
        cols, rows = int(cols), int(rows)
        index = None if index == ALL else int(index)
    except ValueError:
        raise ValueError(
            f"Unknown view {spec!r}: use 'whole', a grid and a tile such as '3x3:5', or every tile with '3x3:all'."
        ) from None
    if not (1 <= cols <= MAX_SIDE and 1 <= rows <= MAX_SIDE):
        raise ValueError(f"Unknown view {spec!r}: grids go from 1x1 to {MAX_SIDE}x{MAX_SIDE}.")
    if index is not None and not 1 <= index <= cols * rows:
        raise ValueError(f"Unknown view {spec!r}: a {cols}x{rows} grid has tiles 1 to {cols * rows}.")
    return cols, rows, index


def tile_box(width: int, height: int, cols: int, rows: int, index: int) -> Box:
    """Where tile `index` sits in a picture of width x height."""
    row, col = divmod(index - 1, cols)
    x0, x1 = round(col * width / cols), round((col + 1) * width / cols)
    y0, y1 = round(row * height / rows), round((row + 1) * height / rows)
    return x0, y0, x1, y1


def crop_tile(frame: np.ndarray, tile: tuple[int, int, int] | None) -> np.ndarray:
    """The part of the frame that shows the chosen tile (the whole frame for None)."""
    if tile is None:
        return frame
    cols, rows, index = tile
    x0, y0, x1, y1 = tile_box(frame.shape[1], frame.shape[0], cols, rows, index)
    return frame[y0:y1, x0:x1]
