import numpy as np
import pytest

from ergotwin.tiles import crop_tile, parse_tile, tile_box


@pytest.mark.parametrize("spec", [None, "", "whole", " WHOLE "])
def test_whole_picture(spec):
    assert parse_tile(spec) is None


@pytest.mark.parametrize("spec, expected", [("3x3:5", (3, 3, 5)), ("2x2:1", (2, 2, 1)), ("3×3:9", (3, 3, 9)),
                                            ("4x2:8", (4, 2, 8))])
def test_grid_and_tile(spec, expected):
    assert parse_tile(spec) == expected


@pytest.mark.parametrize("spec, expected", [("3x3:all", (3, 3, None)), (" 2×2:ALL ", (2, 2, None))])
def test_every_tile_at_once(spec, expected):
    assert parse_tile(spec) == expected


@pytest.mark.parametrize("spec", ["3x3", "3x3:0", "3x3:10", "0x3:1", "7x7:1", "big", "3x3:five", "7x7:all"])
def test_bad_views_are_refused_with_a_reason(spec):
    with pytest.raises(ValueError, match="Unknown view"):
        parse_tile(spec)


def test_tile_boxes_tile_the_picture_exactly():
    boxes = [tile_box(1025, 577, 3, 3, i) for i in range(1, 10)]
    assert boxes[0][:2] == (0, 0) and boxes[-1][2:] == (1025, 577)
    area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes)
    assert area == 1025 * 577  # no gaps, no overlaps


def test_tiles_are_counted_left_to_right_then_top_to_bottom():
    # A 3 x 3 picture where every pixel holds the number of its tile.
    frame = np.repeat(np.repeat(np.arange(1, 10).reshape(3, 3), 4, axis=0), 6, axis=1)
    for index in range(1, 10):
        tile = crop_tile(frame, (3, 3, index))
        assert tile.shape == (4, 6)
        assert (tile == index).all()


def test_whole_picture_is_left_alone():
    frame = np.zeros((576, 1024, 3), np.uint8)
    assert crop_tile(frame, None) is frame


def test_uneven_sizes_cover_the_whole_picture():
    frame = np.zeros((577, 1025, 3), np.uint8)
    widths = [crop_tile(frame, (3, 3, i)).shape[1] for i in (1, 2, 3)]
    heights = [crop_tile(frame, (3, 3, i)).shape[0] for i in (1, 4, 7)]
    assert sum(widths) == 1025 and sum(heights) == 577
