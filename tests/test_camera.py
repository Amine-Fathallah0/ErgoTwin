import math

import numpy as np
import pytest

from ergotwin.angles import measure
from ergotwin.camera import VerticalField
from ergotwin.config import AngleConfig, CameraConfig
from ergotwin.skeleton import INDEX, Person
from tests.figures import side_figure

SIZE = (1280, 720)
HFOV = 60.0
FOCAL = 0.5 / math.tan(math.radians(HFOV) / 2)  # in picture widths


def camera_truth(pitch_deg: float, point) -> tuple[np.ndarray, float]:
    """What a camera looking down by `pitch_deg` does at a pixel: (unit up vector, squash of a standing person)."""
    w, h = SIZE
    p = np.array([(point[0] - w / 2) / w, (point[1] - h / 2) / w])
    theta = math.radians(pitch_deg)
    g = np.array([0.0, math.cos(theta), math.sin(theta)])
    if pitch_deg == 0:
        up = np.array([0.0, -1.0])
    else:
        vanishing_point = np.array([0.0, FOCAL * g[1] / g[2]])
        up = (p - vanishing_point) / np.linalg.norm(p - vanishing_point)
    return up, float(np.linalg.norm(g[:2] - p / FOCAL * g[2]))


def seen_by_camera(person: Person, point, up, squash: float, track_id: int) -> Person:
    """A level-view figure as the tilted camera shows it with its hips at `point`."""
    rel = person.xy - person.xy[INDEX["l_hip"]]
    right = np.array([-up[1], up[0]])
    xy = np.asarray(point, float) + np.outer(rel[:, 0], right) + np.outer(-rel[:, 1] * squash, up)
    box = (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max())
    return Person(track_id, xy, person.conf.copy(), box)


def crowd(pitch_deg: float, rng) -> list[Person]:
    """Forty workers spread over the picture, mostly upright, every fifth one bent well forward."""
    people = []
    for i in range(40):
        point = (rng.uniform(100, 1180), rng.uniform(120, 620))
        up, squash = camera_truth(pitch_deg, point)
        trunk = 55.0 if i % 5 == 0 else rng.uniform(-4.0, 6.0)
        figure = side_figure(trunk=trunk, facing=int(rng.choice([1, -1])), scale=0.4)
        people.append(seen_by_camera(figure, point, up, squash, track_id=i))
    return people


def learn(pitch_deg: float) -> VerticalField:
    rng = np.random.default_rng(1)
    field = VerticalField(CameraConfig(hfov_deg=HFOV))
    for step in range(8):
        field.observe(step * 0.5, crowd(pitch_deg, rng), SIZE, 0.5)
    return field


def test_level_camera_needs_no_correction():
    field = learn(0)
    assert field.pitch_deg < 3
    up, squash = field.frame_at((200, 600), SIZE)
    assert up == pytest.approx([0, -1], abs=0.05)
    assert squash == pytest.approx(1.0, abs=0.04)


@pytest.mark.parametrize("pitch", [25, 40, 55])
def test_learns_how_far_the_camera_looks_down(pitch):
    field = learn(pitch)
    assert field.learned
    assert field.pitch_deg == pytest.approx(pitch, abs=3)
    for point in ((150, 200), (640, 360), (1100, 600)):
        up_true, squash_true = camera_truth(pitch, point)
        up, squash = field.frame_at(point, SIZE)
        assert math.degrees(math.acos(min(1.0, float(up @ up_true)))) < 3
        assert squash == pytest.approx(squash_true, abs=0.05)


def test_nothing_is_corrected_before_enough_people_were_seen():
    field = VerticalField(CameraConfig())
    person = side_figure()
    field.observe(0.0, [person], SIZE, 0.5)
    assert not field.learned
    same, squash = field.rectify(person, SIZE, 0.5)
    assert same is person and squash == 1.0


def test_people_in_one_spot_teach_nothing():
    # One worker who never moves cannot tell a tilted camera from a leaning person.
    field = VerticalField(CameraConfig(hfov_deg=HFOV))
    up, squash = camera_truth(40, (640, 360))
    for step in range(200):
        field.observe(step * 0.5, [seen_by_camera(side_figure(scale=0.4), (640, 360), up, squash, 1)], SIZE, 0.5)
    assert not field.learned


def leaning_crowd(rng, lean_at) -> list[Person]:
    """Forty workers seen by a level camera, legs hidden, each leaning `lean_at(x)` degrees toward image-right."""
    people = []
    for i in range(40):
        point = (rng.uniform(100, 1180), rng.uniform(120, 620))
        figure = side_figure(trunk=lean_at(point[0]), scale=0.4)
        for name in ("l_knee", "r_knee", "l_ankle", "r_ankle"):
            figure.conf[INDEX[name]] = 0.1
        people.append(seen_by_camera(figure, point, np.array([0.0, -1.0]), 1.0, track_id=i))
    return people


def watch(crowd_at_step) -> VerticalField:
    field = VerticalField(CameraConfig(hfov_deg=HFOV))
    for step in range(8):
        field.observe(step * 0.5, crowd_at_step(), SIZE, 0.5)
    return field


def test_everyone_leaning_the_same_way_is_not_a_tilted_camera():
    rng = np.random.default_rng(2)
    assert not watch(lambda: leaning_crowd(rng, lambda x: 25.0)).learned


def test_two_groups_leaning_toward_each_other_are_not_a_tilted_camera():
    rng = np.random.default_rng(2)
    assert not watch(lambda: leaning_crowd(rng, lambda x: 20.0 if x < 640 else -20.0)).learned


def test_the_tilt_can_be_given_instead_of_learned():
    field = VerticalField(CameraConfig(hfov_deg=HFOV, tilt_deg=40.0))
    assert field.learned and field.pitch_deg == pytest.approx(40)
    for point in ((150, 200), (640, 360), (1100, 600)):
        up_true, squash_true = camera_truth(40, point)
        up, squash = field.frame_at(point, SIZE)
        assert up == pytest.approx(up_true, abs=1e-6)
        assert squash == pytest.approx(squash_true, abs=1e-6)


def test_learning_can_be_switched_off():
    rng = np.random.default_rng(1)
    field = VerticalField(CameraConfig(auto_vertical=False))
    for step in range(8):
        field.observe(step * 0.5, crowd(40, rng), SIZE, 0.5)
    assert not field.learned


@pytest.mark.parametrize("lean", [0, 30, 65])
def test_rectified_skeleton_reads_the_true_lean(lean):
    field = learn(40)
    point = (300, 500)  # off-centre, where upright people lean in the picture
    up, squash = camera_truth(40, point)
    seen = seen_by_camera(side_figure(trunk=lean, scale=0.6), point, up, squash, 99)
    upright, undone = field.rectify(seen, SIZE, 0.5)
    assert undone == pytest.approx(squash, abs=0.05)
    assert measure(upright, AngleConfig(), 0.5).trunk_flexion == pytest.approx(lean, abs=4)


def test_without_the_correction_an_upright_worker_off_centre_reads_as_leaning():
    up, squash = camera_truth(40, (300, 500))
    seen = seen_by_camera(side_figure(scale=0.6), (300, 500), up, squash, 99)
    assert abs(measure(seen, AngleConfig(), 0.5).trunk_flexion) > 10
