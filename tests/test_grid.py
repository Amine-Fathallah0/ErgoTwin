"""Every tile of a grid recording at once: each tile tracked alone, everyone drawn back on the grid."""

import numpy as np

from ergotwin import engine
from ergotwin.config import Config
from ergotwin.engine import GridScene
from tests.figures import front_figure


class FakePose:
    """Finds one upright worker in the middle of every tile, always under tracker id 1."""

    device_name = "test"

    def __init__(self, cfg):
        self.cfg = cfg.pose
        self.sizes = []

    def infer_tiles(self, frames):
        self.sizes.append({f.shape for f in frames})
        h, w = frames[0].shape[:2]
        return [[front_figure(origin=(w / 2, h * 0.55), scale=h / 700)] for _ in frames]  # fits inside the tile


def run_grid(monkeypatch, frame, steps=(0.0, 0.6, 1.2, 1.8)):
    drawn = {}

    def render(canvas, people, views, min_conf, privacy):
        drawn.update(canvas=canvas, people=people, views=views)
        return canvas

    monkeypatch.setattr(engine.draw, "render", render)
    cfg = Config()
    scene = GridScene(cfg, FakePose(cfg), "3x3:all")
    for t in steps:
        out = scene.step(frame, t, "12:00:00", privacy=False)
    return scene, out, drawn


def test_every_tile_is_enlarged_to_the_same_size(monkeypatch):
    scene, _, _ = run_grid(monkeypatch, np.zeros((576, 1024, 3), np.uint8))
    assert scene.pose.sizes[-1] == {(612, 1088, 3)}  # one size for all nine: they go in one batch


def test_workers_in_different_tiles_get_different_numbers(monkeypatch):
    _, (picture, workers, flags, camera), _ = run_grid(monkeypatch, np.zeros((576, 1024, 3), np.uint8))
    assert sorted(w["id"] for w in workers) == list(range(1, 10))
    assert sorted(w["tile"] for w in workers) == list(range(1, 10))


def test_people_are_drawn_back_inside_their_own_tile(monkeypatch):
    _, (picture, *_), drawn = run_grid(monkeypatch, np.zeros((576, 1024, 3), np.uint8))
    assert picture.shape == (720, 1280, 3)  # the grid, enlarged to the display width
    cell_w, cell_h = 1280 / 3, 720 / 3
    assert len(drawn["people"]) == 9
    for person in drawn["people"]:
        tile = person.track_id // GridScene.KEY
        row, col = divmod(tile - 1, 3)
        x1, y1, x2, y2 = person.box
        assert col * cell_w <= x1 < x2 <= (col + 1) * cell_w
        assert row * cell_h <= y1 < y2 <= (row + 1) * cell_h
        assert drawn["views"][person.track_id]["tile"] == tile
