import pytest

from ergotwin.angles import measure
from ergotwin.config import AngleConfig, Config
from ergotwin.pipeline import NumberPool, Pipeline
from ergotwin.skeleton import INDEX
from tests.figures import front_figure, side_figure

SIZE = (1280, 720)


def run(pipeline, figure, start, end, step=0.1):
    """Feed the same figure from `start` to `end` seconds. Returns the last frame's views."""
    views = {}
    for i in range(round((end - start) / step) + 1):
        views = pipeline.process(round(start + i * step, 3), [figure], SIZE, clock="12:00:00")
    return views


def test_a_worker_is_shown_once_followed_for_a_moment():
    pipeline = Pipeline(Config())
    assert run(pipeline, side_figure(), 0, 0.5) == {}
    assert list(run(pipeline, side_figure(), 0.6, 1.5)) == [1]


def test_view_has_what_the_page_and_the_teammates_need():
    worker = run(Pipeline(Config()), side_figure(trunk=40, arm=50), 0, 3)[1]
    assert worker["level"] == "caution"
    assert worker["posture"] == "Bent forward 40°"
    assert worker["flagged"] is False
    assert worker["reba"]["trunk"] == 3 and worker["reba"]["upper_arm"] == 3
    assert worker["reba"]["complete"] is True
    back = worker["regions"]["back"]
    assert (back["level"], back["label"], back["angle"], back["pulses"]) == ("caution", "Bent forward 40°", 40, 2)
    assert worker["regions"]["shoulders"]["pulses"] == 1 and worker["regions"]["knees"]["pulses"] == 3


def test_bad_posture_is_flagged_through_the_whole_pipeline():
    pipeline = Pipeline(Config())
    worker = run(pipeline, side_figure(trunk=70), 0, 8)[1]
    assert worker["level"] == "bad" and worker["flagged"] is True
    flags = list(pipeline.flags)
    assert len(flags) == 1
    assert (flags[0]["region"], flags[0]["level"], flags[0]["pulses"]) == ("back", "bad", 2)
    assert flags[0]["message"] == "Worker 1: bent forward 70° for 5 s"


def test_upright_worker_is_safe_and_never_flagged():
    pipeline = Pipeline(Config())
    worker = run(pipeline, side_figure(), 0, 90)[1]
    assert worker["level"] == "safe" and worker["posture"] == "Neutral posture"
    assert list(pipeline.flags) == []


def test_people_with_no_visible_trunk_are_left_out():
    hidden = side_figure()
    for name in ("l_shoulder", "r_shoulder", "l_hip", "r_hip"):
        hidden.conf[INDEX[name]] = 0.1
    assert run(Pipeline(Config()), hidden, 0, 3) == {}


def test_hidden_legs_make_the_whole_body_score_partial():
    figure = side_figure(trunk=30)
    for name in ("l_knee", "r_knee", "l_ankle", "r_ankle"):
        figure.conf[INDEX[name]] = 0.1
    worker = run(Pipeline(Config()), figure, 0, 3)[1]
    assert worker["regions"]["knees"]["level"] is None
    assert worker["regions"]["knees"]["label"] == "Not visible"
    assert worker["reba"]["complete"] is False


def test_sideways_arm_counts_as_abducted_only_from_a_level_camera():
    figure = front_figure(arm_out=60)
    assert measure(figure, AngleConfig(), 0.5).arm_abducted == {"l": True, "r": True}
    assert measure(figure, AngleConfig(), 0.5, tilted=True).arm_abducted == {"l": False, "r": False}


def test_workers_get_small_numbers_that_are_reused_after_they_leave():
    pipeline = Pipeline(Config())

    def tracked(track_id, x):
        figure = side_figure(origin=(x, 500.0))
        figure.track_id = track_id
        return figure

    views = {}
    for i in range(21):  # tracker ids 40 and 97 arrive together
        views = pipeline.process(i * 0.1, [tracked(40, 300.0), tracked(97, 700.0)], SIZE)
    assert {track: view["id"] for track, view in views.items()} == {40: 1, 97: 2}

    for i in range(21, 80):  # 40 leaves; 97 stays
        views = pipeline.process(i * 0.1, [tracked(97, 700.0)], SIZE)
    assert views[97]["id"] == 2

    for i in range(80, 100):  # a new tracker id takes the number that 40 left free
        views = pipeline.process(i * 0.1, [tracked(97, 700.0), tracked(205, 300.0)], SIZE)
    assert {track: view["id"] for track, view in views.items()} == {97: 2, 205: 1}


def test_a_worker_who_left_is_forgotten():
    pipeline = Pipeline(Config())
    run(pipeline, side_figure(trunk=70), 0, 3)
    assert pipeline.process(10.0, [], SIZE) == {}
    assert pipeline.monitor.workers == {}
    assert run(pipeline, side_figure(trunk=70), 10.1, 12)[1]["held_s"] == pytest.approx(0.9, abs=0.3)


def test_tiles_of_a_grid_share_one_set_of_worker_numbers():
    numbers = NumberPool()
    left, right = Pipeline(Config(), numbers, tile=1), Pipeline(Config(), numbers, tile=2)
    a = run(left, side_figure(), 0, 1.5)[1]
    b = run(right, side_figure(), 0, 1.5)[1]  # the same tracker id, in another tile
    assert (a["id"], a["tile"]) == (1, 1)
    assert (b["id"], b["tile"]) == (2, 2)


def test_a_flag_says_which_tile_it_came_from():
    pipeline = Pipeline(Config(), tile=4)
    run(pipeline, side_figure(trunk=70), 0, 8)
    assert pipeline.flags and all(flag["tile"] == 4 for flag in pipeline.flags)


def test_number_pool_hands_out_the_lowest_free_number():
    numbers = NumberPool()
    assert [numbers.take() for _ in range(3)] == [1, 2, 3]
    numbers.release({2})
    assert numbers.take() == 2 and numbers.take() == 4
