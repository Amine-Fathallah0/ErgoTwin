import pytest

from ergotwin.angles import Angles
from ergotwin.config import AngleConfig, Config, StationConfig
from ergotwin.reba import (
    TABLE_A, TABLE_B, TABLE_C, legs_score, lower_arm_score, neck_score, region_levels, risk_level,
    score, trunk_score, upper_arm_score,
)

CFG = AngleConfig()


def posture(trunk=0.0, neck=5.0, arm=5.0, elbow=80.0, knee=5.0, side_bend=0.0) -> Angles:
    return Angles(
        view="side", trunk_flexion=trunk, trunk_side_bend=side_bend, neck_flexion=neck, neck_side_bend=0.0,
        upper_arm={"l": arm, "r": arm}, elbow={"l": elbow, "r": elbow}, knee={"l": knee, "r": knee},
    )


@pytest.mark.parametrize("angle, expected", [(0, 1), (5, 1), (10, 2), (20, 2), (21, 3), (60, 3), (61, 4),
                                             (-10, 2), (-20, 2), (-25, 3)])
def test_trunk_bands(angle, expected):
    assert trunk_score(angle, 0.0, CFG) == expected


def test_trunk_side_bend_adds_one():
    assert trunk_score(30, 20.0, CFG) == 4
    assert trunk_score(0, 20.0, CFG) == 2


@pytest.mark.parametrize("angle, expected", [(0, 1), (20, 1), (21, 2), (-10, 1), (-20, 2)])
def test_neck_bands(angle, expected):
    assert neck_score(angle, 0.0, CFG) == expected


def test_neck_side_bend_adds_one():
    assert neck_score(30, 30.0, CFG) == 3
    assert neck_score(30, 20.0, CFG) == 2  # small sideways tilts are within the noise


@pytest.mark.parametrize("bend, expected", [(0, 1), (29, 1), (30, 2), (60, 2), (61, 3)])
def test_leg_bands(bend, expected):
    assert legs_score({"l": bend, "r": bend}, seated=False) == expected


def test_one_bent_knee_is_walking_not_crouching():
    assert legs_score({"l": 50.0, "r": 2.0}, seated=False) == 1


def test_a_hidden_leg_leaves_the_visible_knee_to_decide():
    assert legs_score({"l": 70.0, "r": None}, seated=False) == 3


def test_seated_work_does_not_penalise_bent_knees():
    assert legs_score({"l": 90.0, "r": 90.0}, seated=True) == 1


@pytest.mark.parametrize("lift, expected", [(0, 1), (20, 1), (21, 2), (45, 2), (46, 3), (90, 3), (91, 4),
                                            (-20, 1), (-25, 2)])
def test_upper_arm_bands(lift, expected):
    assert upper_arm_score(lift, abducted=False, supported=False) == expected


def test_upper_arm_adjustments():
    assert upper_arm_score(50, abducted=True, supported=False) == 4
    assert upper_arm_score(50, abducted=False, supported=True) == 2
    assert upper_arm_score(10, abducted=False, supported=True) == 1  # never below 1


@pytest.mark.parametrize("elbow, expected", [(59, 2), (60, 1), (100, 1), (101, 2), (0, 2)])
def test_lower_arm_bands(elbow, expected):
    assert lower_arm_score(elbow) == expected


def test_table_shapes_and_corners():
    assert len(TABLE_A) == 5 and all(len(row) == 3 and all(len(c) == 4 for c in row) for row in TABLE_A)
    assert len(TABLE_B) == 6 and all(len(row) == 2 and all(len(c) == 3 for c in row) for row in TABLE_B)
    assert len(TABLE_C) == 12 and all(len(row) == 12 for row in TABLE_C)
    assert TABLE_A[0][0][0] == 1 and TABLE_A[4][2][3] == 9
    assert TABLE_B[0][0][0] == 1 and TABLE_B[5][1][2] == 9
    assert TABLE_C[0][0] == 1 and TABLE_C[11][11] == 12


def test_neutral_posture_scores_one():
    r = score(posture(), StationConfig(), CFG)
    assert (r.trunk, r.neck, r.legs, r.upper_arm, r.lower_arm, r.wrist) == (1, 1, 1, 1, 1, 1)
    assert r.score == 1 and r.risk == "negligible"


def test_stooped_lift_of_a_heavy_load():
    # Trunk 70 -> 4, neck 25 -> 2, legs 1: Table A = 5, +2 for a 12 kg load = 7.
    # Upper arm 50 -> 3, straight elbow -> 2, wrist 1: Table B = 4, +1 fair grip = 5.
    # Table C (7, 5) = 9, no activity point: REBA 9, high risk.
    station = StationConfig(load_kg=12, coupling=1)
    r = score(posture(trunk=70, neck=25, arm=50, elbow=30), station, CFG)
    assert (r.trunk, r.neck, r.legs, r.upper_arm, r.lower_arm) == (4, 2, 1, 3, 2)
    assert (r.score_a, r.score_b, r.score_c) == (7, 5, 9)
    assert r.score == 9 and r.risk == "high"


def test_a_posture_held_over_a_minute_adds_one():
    moving = score(posture(trunk=40), StationConfig(), CFG, static=False)
    held = score(posture(trunk=40), StationConfig(), CFG, static=True)
    assert held.score == moving.score + 1 and held.activity == 1


def test_the_more_loaded_arm_is_scored():
    angles = posture()
    angles.upper_arm = {"l": 10.0, "r": 100.0}
    assert score(angles, StationConfig(), CFG).upper_arm == 4


def test_hidden_joints_score_as_neutral():
    r = score(Angles(), StationConfig(), CFG)
    assert r.score == 1


@pytest.mark.parametrize("value, expected", [(1, "negligible"), (2, "low"), (3, "low"), (4, "medium"),
                                             (7, "medium"), (8, "high"), (10, "high"), (11, "very high"),
                                             (15, "very high")])
def test_risk_bands(value, expected):
    assert risk_level(value) == expected


def test_region_levels_follow_the_part_scores():
    cuts = Config().levels
    r = score(posture(trunk=70, arm=60, knee=40), StationConfig(), CFG)
    assert region_levels(r, posture(trunk=70, arm=60, knee=40), cuts) == {
        "back": 2, "neck": 0, "shoulders": 1, "knees": 1,
    }


def test_regions_that_are_not_visible_have_no_level():
    r = score(Angles(), StationConfig(), CFG)
    assert region_levels(r, Angles(), Config().levels) == {
        "back": None, "neck": None, "shoulders": None, "knees": None,
    }
