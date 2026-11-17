"""Stick figures with known joint angles, for testing the angle code.

Figures are built in a frame where y points up and the worker faces +x, then
converted to image pixels (y down).
"""

from __future__ import annotations

import math

import numpy as np

from ergotwin.skeleton import INDEX, KEYPOINTS, Person

TORSO, NECK, NOSE, UPPER_ARM, FOREARM, THIGH, SHANK = 100.0, 40.0, 22.0, 60.0, 50.0, 85.0, 85.0
EAR_LEAN = 10.0  # the ear's natural forward lean from the shoulder, in degrees


def _tilt(deg: float) -> np.ndarray:
    """Unit vector tilted from straight up toward the front by `deg`."""
    a = math.radians(deg)
    return np.array([math.sin(a), math.cos(a)])


def _person(points: dict, origin, scale: float, conf: float, facing: int = 1) -> Person:
    xy = np.zeros((len(KEYPOINTS), 2))
    for name, pt in points.items():
        xy[INDEX[name]] = (origin[0] + facing * pt[0] * scale, origin[1] - pt[1] * scale)
    box = (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max())
    return Person(track_id=1, xy=xy, conf=np.full(len(KEYPOINTS), conf), box=box)


def side_figure(trunk=0.0, neck=0.0, arm=0.0, elbow=0.0, knee=0.0, facing=1,
                origin=(400.0, 500.0), scale=1.0, conf=0.9) -> Person:
    """A worker seen from the side. Angles in degrees; facing=+1 looks toward image-right."""
    hip = np.zeros(2)
    shoulder = hip + TORSO * _tilt(trunk)
    ear = shoulder + NECK * _tilt(trunk + neck + EAR_LEAN)
    nose = ear + NOSE * _tilt(trunk + neck + 90.0)
    elbow_pt = shoulder - UPPER_ARM * _tilt(trunk - arm)
    wrist = elbow_pt - FOREARM * _tilt(trunk - arm - elbow)
    half = math.radians(knee / 2.0)  # thigh swings forward and shank back by half the bend each
    knee_pt = hip + THIGH * np.array([math.sin(half), -math.cos(half)])
    ankle = knee_pt + SHANK * np.array([-math.sin(half), -math.cos(half)])

    points = {"nose": nose, "l_eye": nose, "r_eye": nose, "l_ear": ear, "r_ear": ear}
    for s in ("l", "r"):  # both sides overlap in a side view
        points |= {f"{s}_shoulder": shoulder, f"{s}_elbow": elbow_pt, f"{s}_wrist": wrist,
                   f"{s}_hip": hip, f"{s}_knee": knee_pt, f"{s}_ankle": ankle}
    return _person(points, origin, scale, conf, facing)


def front_figure(bend_toward_camera=0.0, side_bend=0.0, arm_out=0.0,
                 origin=(400.0, 500.0), scale=1.0, conf=0.9) -> Person:
    """A worker facing the camera, legs straight. Their left side appears on image-right."""
    f, s, a = math.radians(bend_toward_camera), math.radians(side_bend), math.radians(arm_out)
    hip = np.zeros(2)
    shoulder = hip + TORSO * np.array([math.sin(s), math.cos(s) * math.cos(f)])
    head = shoulder + NECK * np.array([0.0, math.cos(f)])

    points = {"nose": head + np.array([0.0, -4.0])}
    for side, sx in (("l", 1.0), ("r", -1.0)):
        shoulder_s = shoulder + np.array([sx * 37.0, 0.0])
        hip_s = hip + np.array([sx * 17.0, 0.0])
        arm_dir = np.array([sx * math.sin(a), -math.cos(a)])
        elbow_pt = shoulder_s + UPPER_ARM * arm_dir
        knee_pt = hip_s + np.array([0.0, -THIGH])
        points |= {
            f"{side}_eye": head + np.array([sx * 4.0, 2.0]),
            f"{side}_ear": head + np.array([sx * 8.0, 0.0]),
            f"{side}_shoulder": shoulder_s,
            f"{side}_elbow": elbow_pt,
            f"{side}_wrist": elbow_pt + FOREARM * arm_dir,
            f"{side}_hip": hip_s,
            f"{side}_knee": knee_pt,
            f"{side}_ankle": knee_pt + np.array([0.0, -SHANK]),
        }
    return _person(points, origin, scale, conf)
