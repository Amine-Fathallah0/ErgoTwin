"""Joint angles from 2D keypoints.

Angles are in degrees and measured in the image, so they are most accurate
when the camera sees the worker from the side or at an angle. Image
coordinates: x to the right, y down. The camera is assumed to be roughly level.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import AngleConfig
from .skeleton import Person

SIDES = ("l", "r")
DOWN = np.array([0.0, 1.0])


def _length(v) -> float:
    return float(math.hypot(v[0], v[1]))


def angle_between(u, v) -> float:
    """Angle between two vectors, 0 to 180."""
    lu, lv = _length(u), _length(v)
    if lu < 1e-9 or lv < 1e-9:
        return 0.0
    cos = float(u[0] * v[0] + u[1] * v[1]) / (lu * lv)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def joint_flexion(a, b, c) -> float:
    """Bend at joint b between points a and c: 0 when the limb is straight."""
    return 180.0 - angle_between(a - b, c - b)


def _cross(u, v) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def _mid(a, b):
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a if a is not None else b


def _pair() -> dict:
    return {"l": None, "r": None}


@dataclass
class Angles:
    """One worker's posture in one frame. None means the joints were not visible."""

    view: str = "unknown"  # how the camera sees the worker: "side", "oblique" or "front"
    trunk_flexion: float | None = None  # forward bend from vertical; negative = leaning back
    trunk_side_bend: float | None = None
    neck_flexion: float | None = None  # relative to the trunk; negative = tilted back
    neck_side_bend: float | None = None
    upper_arm: dict = field(default_factory=_pair)  # lift away from the trunk; negative = behind the body
    arm_abducted: dict = field(default_factory=lambda: {"l": False, "r": False})
    elbow: dict = field(default_factory=_pair)  # 0 = straight arm
    knee: dict = field(default_factory=_pair)  # 0 = straight leg


def trunk_points(person: Person, min_conf: float):
    """Shoulder and hip points of the trunk line: the body midline if both sides show, else one side."""
    ls, rs = person.pt("l_shoulder", min_conf), person.pt("r_shoulder", min_conf)
    lh, rh = person.pt("l_hip", min_conf), person.pt("r_hip", min_conf)
    if ls is not None and rs is not None and lh is not None and rh is not None:
        return (ls + rs) / 2.0, (lh + rh) / 2.0
    if ls is not None and lh is not None:
        return ls, lh
    if rs is not None and rh is not None:
        return rs, rh
    return _mid(ls, rs), _mid(lh, rh)


def measure(person: Person, cfg: AngleConfig, min_conf: float, torso_ref: float | None = None,
            tilted: bool = False) -> Angles:
    """All joint angles of one person.

    torso_ref is the torso's full length in pixels (see TorsoReference). With
    it, a torso that looks short is read as a bend toward or away from the camera.
    tilted says the skeleton comes from a camera that looks down: an arm lifted
    sideways then looks the same as an arm reaching away, so it is not called abducted.
    """

    def p(name: str):
        return person.pt(name, min_conf)

    out = Angles()
    x1, y1, x2, y2 = person.box
    if max(x2 - x1, y2 - y1) < cfg.min_person_px:
        return out  # too small in the picture to measure anything
    shoulder, hip = trunk_points(person, min_conf)
    trunk = None
    if shoulder is not None and hip is not None and _length(shoulder - hip) > 1e-6:
        trunk = shoulder - hip

    facing = 0
    if trunk is not None:
        facing = _measure_trunk(out, trunk, shoulder, p, cfg, torso_ref)
        _measure_neck(out, trunk, shoulder, _mid(p("l_ear"), p("r_ear")), facing, cfg)

    for side in SIDES:
        _measure_arm(out, side, p, trunk, facing, cfg, torso_ref, tilted)
        hip_s, knee, ankle = p(f"{side}_hip"), p(f"{side}_knee"), p(f"{side}_ankle")
        if hip_s is not None and knee is not None and ankle is not None:
            out.knee[side] = joint_flexion(hip_s, knee, ankle)
    return out


def _facing(nose, ear, shoulder, scale: float) -> int:
    """Which way the worker's front points: +1 toward image-right, -1 toward image-left, 0 if unclear.

    The nose sits in front of the line from shoulder to ear, however the trunk
    and neck are bent. Facing the camera or away from it, the nose is on that
    line and the answer is 0.
    """
    if nose is None or scale <= 0:
        return 0
    if ear is not None and shoulder is not None and _length(ear - shoulder) > 1e-6:
        neck = ear - shoulder
        offset = _cross(neck, nose - ear) / _length(neck)
    elif shoulder is not None:
        offset = float(nose[0] - shoulder[0])
    else:
        return 0
    if abs(offset) < 0.08 * scale:
        return 0
    return 1 if offset > 0 else -1


def _measure_trunk(out: Angles, trunk, shoulder, p, cfg: AngleConfig, torso_ref: float | None) -> int:
    """Fills the view and the trunk angles. Returns the facing direction (see _facing)."""
    apparent = _length(trunk)
    full = max(torso_ref, apparent) if torso_ref else apparent
    up = -trunk[1]  # how far the shoulders sit above the hips
    lean = math.degrees(math.atan2(abs(trunk[0]), up))  # tilt from vertical as the image shows it

    ls, rs = p("l_shoulder"), p("r_shoulder")
    if ls is not None and rs is not None:
        # Sideways distance only: seen from above, a side-on worker's far shoulder sits
        # higher in the picture than the near one, which is depth, not width.
        ratio = abs(float(ls[0] - rs[0])) / full
        if ratio >= cfg.front_view_ratio:
            out.view = "front"
        elif ratio <= cfg.side_view_ratio:
            out.view = "side"
        else:
            out.view = "oblique"
    else:
        out.view = "side"  # the far shoulder is hidden behind the body

    # Depth: how far the trunk tilts toward or away from the camera, read from how short it looks.
    depth = 0.0
    if cfg.foreshortening and torso_ref and apparent < full:
        d = math.sqrt(full * full - apparent * apparent)
        if math.degrees(math.atan2(d, up)) >= cfg.foreshortening_min_deg:
            depth = d

    if out.view == "front":
        # Facing the camera, a tilt in the image is a sideways bend and depth is the forward bend.
        out.trunk_flexion = math.degrees(math.atan2(depth, up)) if depth else 0.0
        out.trunk_side_bend = math.degrees(math.atan2(abs(trunk[0]), math.hypot(up, depth)))
        return 0

    facing = _facing(p("nose"), _mid(p("l_ear"), p("r_ear")), shoulder, full)
    if lean > 45.0:  # a bend this deep is forward, whatever the head shows
        facing = 1 if trunk[0] > 0 else -1
    total = math.degrees(math.atan2(math.hypot(trunk[0], depth), up))
    # Leaning back is rare at work, so it is only reported from a clean side view.
    leaning_back = (
        out.view == "side"
        and facing != 0
        and depth == 0.0
        and lean > cfg.upright_tol_deg
        and (trunk[0] > 0) != (facing > 0)
    )
    out.trunk_flexion = -total if leaning_back else total
    out.trunk_side_bend = 0.0
    return facing


def _measure_neck(out: Angles, trunk, shoulder, ear, facing: int, cfg: AngleConfig) -> None:
    if ear is None or _length(ear - shoulder) < max(1e-6, cfg.min_neck_px):
        return  # hidden, or too short in the picture for its angle to mean anything
    neck = ear - shoulder
    angle = angle_between(neck, trunk)
    if out.view == "front":
        out.neck_side_bend = angle  # a nod toward the camera cannot be read from this view
        return
    out.neck_side_bend = 0.0
    # _cross > 0 means the head leans toward image-right of the trunk line.
    forward = facing == 0 or (_cross(trunk, neck) > 0) == (facing > 0)
    if out.view == "side" and facing != 0:
        out.neck_flexion = (angle if forward else -angle) - cfg.neck_neutral_deg
    elif forward:
        out.neck_flexion = max(0.0, angle - cfg.neck_neutral_deg)
    else:
        # Seen at an angle, a head that seems to lean back is mostly the turn of the body.
        out.neck_flexion = 0.0


def _measure_arm(out: Angles, side: str, p, trunk, facing: int, cfg: AngleConfig, torso_ref: float | None,
                 tilted: bool) -> None:
    shoulder, elbow, wrist = p(f"{side}_shoulder"), p(f"{side}_elbow"), p(f"{side}_wrist")
    if shoulder is None or elbow is None:
        return
    # The arm is measured against the trunk line, not the same-side hip: shoulders are
    # wider than hips, which would read as a lifted arm when the worker faces the camera.
    hip = p(f"{side}_hip")
    if trunk is not None:
        down = -trunk
    elif hip is not None:
        down = hip - shoulder
    else:
        down = DOWN
    # A torso pointing at the camera gives no usable direction: measure from straight down instead.
    if _length(down) < 1e-6 or (torso_ref and _length(down) < 0.4 * torso_ref):
        down = DOWN

    arm = elbow - shoulder
    lift = angle_between(arm, down)
    if out.view == "front":
        out.arm_abducted[side] = lift > cfg.abduction_deg and not tilted
    elif facing != 0 and out.view == "side":
        forward = (_cross(down, arm) < 0) == (facing > 0)
        if not forward:
            lift = -lift
    out.upper_arm[side] = lift
    if wrist is not None:
        out.elbow[side] = joint_flexion(shoulder, elbow, wrist)


def leg_points(person: Person, min_conf: float):
    """The longer-looking leg as (length in pixels, hip, ankle, is it straight), or None if no whole leg shows."""
    best = None
    for side in SIDES:
        hip = person.pt(f"{side}_hip", min_conf)
        knee = person.pt(f"{side}_knee", min_conf)
        ankle = person.pt(f"{side}_ankle", min_conf)
        if hip is None or knee is None or ankle is None:
            continue
        thigh, shank = _length(knee - hip), _length(ankle - knee)
        if thigh < 1e-6 or shank < 1e-6:
            continue
        total = thigh + shank
        if best is None or total > best[0]:
            # A thigh pointing at the camera looks short next to the shank: not a usable leg.
            straight = joint_flexion(hip, knee, ankle) < 25.0 and 0.7 < thigh / shank < 1.4
            best = (total, hip, ankle, straight)
    return best


class TorsoReference:
    """Learns how long one worker's torso is next to their legs.

    A flat image cannot show a bend toward the camera as an angle: the torso
    just looks shorter. Legs give the scale (they stay the same length as the
    worker moves nearer or farther), so once the worker's own torso-to-leg
    ratio is known, a short-looking torso can be turned into a bend angle.
    """

    def __init__(self, min_samples: int = 10, sample_every_s: float = 0.2, window: int = 900,
                 ratio_bounds: tuple[float, float] = (0.45, 0.85)):
        self.min_samples = min_samples
        self.sample_every_s = sample_every_s
        self.ratio_bounds = ratio_bounds
        self._ratios: deque[float] = deque(maxlen=window)
        self._last_sample = -math.inf

    def update(self, t: float, person: Person, min_conf: float) -> float | None:
        """Feed one frame. Returns the torso's full length in pixels at the worker's
        current distance, or None while it is not known."""
        found = leg_points(person, min_conf)
        if found is None:
            return None
        leg, _, _, straight = found
        shoulder, hip = trunk_points(person, min_conf)
        if shoulder is not None and hip is not None and straight and t - self._last_sample >= self.sample_every_s:
            self._ratios.append(_length(shoulder - hip) / leg)
            self._last_sample = t
        if len(self._ratios) < self.min_samples:
            return None
        # The torso looks longest when upright, so a high percentile is the upright ratio.
        ratio = float(np.percentile(self._ratios, 85))
        lo, hi = self.ratio_bounds
        return min(hi, max(lo, ratio)) * leg


class AngleSmoother:
    """Exponential smoothing of one worker's angles, to calm frame-to-frame jitter."""

    def __init__(self, alpha: float):
        self.alpha = alpha  # weight of the newest value
        self._prev: dict[str, float] = {}

    def apply(self, a: Angles) -> Angles:
        a.trunk_flexion = self._mix("trunk", a.trunk_flexion)
        a.trunk_side_bend = self._mix("trunk_side", a.trunk_side_bend)
        a.neck_flexion = self._mix("neck", a.neck_flexion)
        a.neck_side_bend = self._mix("neck_side", a.neck_side_bend)
        for side in SIDES:
            a.upper_arm[side] = self._mix(f"arm_{side}", a.upper_arm[side])
            a.elbow[side] = self._mix(f"elbow_{side}", a.elbow[side])
            a.knee[side] = self._mix(f"knee_{side}", a.knee[side])
        return a

    def _mix(self, key: str, value: float | None) -> float | None:
        if value is None:
            self._prev.pop(key, None)
            return None
        prev = self._prev.get(key)
        new = value if prev is None else self.alpha * value + (1.0 - self.alpha) * prev
        self._prev[key] = new
        return new
