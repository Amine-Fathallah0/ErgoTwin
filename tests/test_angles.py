import pytest

from ergotwin.angles import AngleSmoother, Angles, TorsoReference, angle_between, joint_flexion, measure
from ergotwin.config import AngleConfig
from ergotwin.skeleton import INDEX
from tests.figures import TORSO, front_figure, side_figure

import numpy as np

CFG = AngleConfig()
TOL = 1.0  # degrees


def m(person, torso_ref=None):
    return measure(person, CFG, 0.5, torso_ref)


def learned_reference(scale=1.0):
    """A torso reference that has watched the worker stand upright for a few seconds."""
    ref = TorsoReference()
    for i in range(15):
        ref.update(i * 0.25, front_figure(scale=scale), 0.5)
    return ref


# --- helpers -----------------------------------------------------------------

def test_angle_between_and_joint_flexion():
    assert angle_between(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(90)
    a, b, c = np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 2.0])
    assert joint_flexion(a, b, c) == pytest.approx(0)  # straight limb
    assert joint_flexion(a, b, np.array([1.0, 1.0])) == pytest.approx(90)


# --- side view ---------------------------------------------------------------

def test_upright_side_view_reads_neutral():
    a = m(side_figure())
    assert a.view == "side"
    assert a.trunk_flexion == pytest.approx(0, abs=TOL)
    assert a.neck_flexion == pytest.approx(0, abs=TOL)
    for s in ("l", "r"):
        assert a.upper_arm[s] == pytest.approx(0, abs=TOL)
        assert a.elbow[s] == pytest.approx(0, abs=TOL)
        assert a.knee[s] == pytest.approx(0, abs=TOL)


@pytest.mark.parametrize("facing", [1, -1])
@pytest.mark.parametrize("deg", [20, 45, 70, 110])
def test_forward_bend_seen_from_the_side(deg, facing):
    assert m(side_figure(trunk=deg, facing=facing)).trunk_flexion == pytest.approx(deg, abs=TOL)


@pytest.mark.parametrize("facing", [1, -1])
def test_leaning_back_is_negative(facing):
    assert m(side_figure(trunk=-25, facing=facing)).trunk_flexion == pytest.approx(-25, abs=TOL)


@pytest.mark.parametrize("facing", [1, -1])
def test_neck_bend_is_relative_to_the_trunk(facing):
    assert m(side_figure(neck=30, facing=facing)).neck_flexion == pytest.approx(30, abs=TOL)
    assert m(side_figure(trunk=40, neck=30, facing=facing)).neck_flexion == pytest.approx(30, abs=TOL)
    # Head in line with a bent trunk: the neck itself is neutral.
    assert m(side_figure(trunk=70, facing=facing)).neck_flexion == pytest.approx(0, abs=TOL)


def test_neck_tilted_back_is_negative():
    assert m(side_figure(neck=-30)).neck_flexion == pytest.approx(-30, abs=TOL)


@pytest.mark.parametrize("facing", [1, -1])
@pytest.mark.parametrize("deg", [30, 60, 120])
def test_arm_lift_seen_from_the_side(deg, facing):
    a = m(side_figure(arm=deg, facing=facing))
    assert a.upper_arm["l"] == pytest.approx(deg, abs=TOL)
    assert a.upper_arm["r"] == pytest.approx(deg, abs=TOL)


def test_arm_behind_the_body_is_negative():
    assert m(side_figure(arm=-30)).upper_arm["l"] == pytest.approx(-30, abs=TOL)


def test_arm_lift_is_relative_to_a_bent_trunk():
    assert m(side_figure(trunk=60, arm=30)).upper_arm["l"] == pytest.approx(30, abs=TOL)


def test_elbow_and_knee_bend():
    a = m(side_figure(elbow=90, knee=75))
    assert a.elbow["l"] == pytest.approx(90, abs=TOL)
    assert a.knee["r"] == pytest.approx(75, abs=TOL)


def test_unsure_joints_are_left_out():
    person = side_figure(knee=80)
    person.conf[INDEX["l_knee"]] = 0.1
    person.conf[INDEX["r_knee"]] = 0.1
    a = m(person)
    assert a.knee == {"l": None, "r": None}
    assert a.trunk_flexion is not None


def test_people_too_small_in_the_picture_are_not_measured():
    a = m(side_figure(trunk=40, scale=0.1))  # about 30 pixels tall
    assert a.trunk_flexion is None
    assert a.upper_arm == {"l": None, "r": None}


def test_a_neck_too_short_in_the_picture_is_not_measured():
    a = m(side_figure(neck=40, scale=0.3))  # the neck is 12 pixels long
    assert a.neck_flexion is None
    assert a.trunk_flexion is not None


# --- front view --------------------------------------------------------------

def test_front_view_upright():
    a = m(front_figure())
    assert a.view == "front"
    assert a.trunk_flexion == pytest.approx(0, abs=TOL)
    assert a.trunk_side_bend == pytest.approx(0, abs=TOL)
    assert a.upper_arm["l"] == pytest.approx(0, abs=TOL)
    assert a.arm_abducted == {"l": False, "r": False}


def test_front_view_tilt_is_a_side_bend():
    a = m(front_figure(side_bend=25))
    assert a.trunk_side_bend == pytest.approx(25, abs=TOL)
    assert a.trunk_flexion == pytest.approx(0, abs=TOL)


def test_front_view_arm_out_is_abduction():
    a = m(front_figure(arm_out=60))
    assert a.upper_arm["l"] == pytest.approx(60, abs=TOL)
    assert a.upper_arm["r"] == pytest.approx(60, abs=TOL)
    assert a.arm_abducted == {"l": True, "r": True}


def test_bend_toward_camera_is_invisible_without_a_reference():
    assert m(front_figure(bend_toward_camera=60)).trunk_flexion == pytest.approx(0, abs=TOL)


def test_seen_at_an_angle_a_backward_head_lean_is_not_reported():
    person = front_figure()
    for name, dx in (("l_shoulder", -15.0), ("r_shoulder", 15.0),  # narrower shoulders: body half turned
                     ("nose", 15.0),  # face turned toward image-right
                     ("l_ear", -14.0), ("r_ear", -14.0)):  # head sits toward image-left
        person.xy[INDEX[name]][0] += dx
    a = m(person)
    assert a.view == "oblique"
    assert a.neck_flexion == pytest.approx(0, abs=TOL)
    assert a.trunk_flexion >= 0


# --- torso reference (bends toward the camera) -------------------------------

def test_reference_learns_the_torso_length():
    ref = learned_reference()
    assert ref.update(10.0, front_figure(), 0.5) == pytest.approx(TORSO, rel=0.01)


def test_reference_needs_a_few_upright_frames_first():
    ref = TorsoReference()
    assert ref.update(0.0, front_figure(), 0.5) is None


@pytest.mark.parametrize("deg", [30, 60, 80])
def test_bend_toward_camera_with_a_reference(deg):
    ref = learned_reference()
    person = front_figure(bend_toward_camera=deg)
    a = m(person, ref.update(10.0, person, 0.5))
    assert a.view == "front"
    assert a.trunk_flexion == pytest.approx(deg, abs=TOL)
    assert a.trunk_side_bend == pytest.approx(0, abs=TOL)


def test_small_bend_toward_camera_is_ignored_as_noise():
    ref = learned_reference()
    person = front_figure(bend_toward_camera=15)
    assert m(person, ref.update(10.0, person, 0.5)).trunk_flexion == pytest.approx(0, abs=TOL)


def test_walking_away_from_the_camera_is_not_a_bend():
    ref = learned_reference(scale=1.0)
    far = front_figure(scale=0.5)  # same posture, half the size in the picture
    assert m(far, ref.update(10.0, far, 0.5)).trunk_flexion == pytest.approx(0, abs=TOL)


def test_reference_ignores_frames_with_bent_legs():
    ref = TorsoReference()
    for i in range(30):
        assert ref.update(i * 0.25, side_figure(knee=90), 0.5) is None


def test_side_view_bend_is_unchanged_by_a_reference():
    ref = learned_reference()
    person = side_figure(trunk=50)
    assert m(person, ref.update(10.0, person, 0.5)).trunk_flexion == pytest.approx(50, abs=TOL)


# --- smoothing ---------------------------------------------------------------

def test_smoother_averages_and_forgets_hidden_joints():
    smoother = AngleSmoother(alpha=0.5)
    assert smoother.apply(Angles(trunk_flexion=0.0)).trunk_flexion == pytest.approx(0)
    assert smoother.apply(Angles(trunk_flexion=10.0)).trunk_flexion == pytest.approx(5)
    assert smoother.apply(Angles(trunk_flexion=None)).trunk_flexion is None
    assert smoother.apply(Angles(trunk_flexion=40.0)).trunk_flexion == pytest.approx(40)
