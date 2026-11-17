"""Learns what "upright" looks like to this camera, from the workers themselves.

A ceiling camera looks down, so people do not stand straight up in the picture.
They point away from one spot below it (the vanishing point of vertical lines),
and they look squashed from head to foot. This module finds that spot from the
body lines of the people seen so far, then turns each skeleton into what a
level camera at the same place would show. A level camera needs no correction
and gets none.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .angles import leg_points, trunk_points
from .config import CameraConfig
from .skeleton import Person

UP = np.array([0.0, -1.0])
SAMPLE_EVERY_S = 0.5  # one body line per person this often
REFIT_EVERY_S = 2.0
MIN_SAMPLES = 80
MAX_SAMPLES = 4000
MIN_PLACES = 4  # body lines must come from at least this many different parts of the picture
MAX_ROLL_DEG = 10.0  # a camera is taken to be mounted nearly upright
FAR_AWAY = 50.0  # a meeting point this many picture widths away means a level camera
MIN_SQUASH = 0.35  # looking almost straight down: stop stretching, the view is too steep to trust
LEVEL_SQUASH = math.cos(math.radians(10.0))  # above this the camera counts as level


def _unit_rows(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


class VerticalField:
    """Which way is up, and how squashed a standing person looks, at each place in the picture."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.learned = False
        self.pitch_deg = 0.0  # how far the camera looks down at the centre of the picture
        self._samples: deque[tuple[float, float, float, float, float]] = deque(maxlen=MAX_SAMPLES)
        self._last_sample: dict[int, float] = {}
        self._last_fit = -math.inf
        self._fresh = 0
        self._vp: np.ndarray | None = None  # vanishing point of verticals, homogeneous, normalised coordinates
        self._sign = 1.0
        self._gravity = np.array([0.0, 1.0, 0.0])
        self._focal = 0.5 / math.tan(math.radians(cfg.hfov_deg) / 2.0)  # in picture widths
        self._aspect = 9.0 / 16.0  # picture height / width, updated from the frames seen
        if cfg.tilt_deg > 0:  # the installer knows the tilt: use it instead of learning
            theta = math.radians(min(cfg.tilt_deg, 85.0))
            self._gravity = np.array([0.0, math.cos(theta), math.sin(theta)])
            self._vp = np.array([0.0, self._gravity[1], self._gravity[2] / self._focal])
            self.pitch_deg = math.degrees(theta)
            self.learned = True

    # --- learning ------------------------------------------------------------

    def observe(self, t: float, people: list[Person], size: tuple[int, int], min_conf: float) -> None:
        """Collect body lines from this frame and refit now and then."""
        if not self.cfg.auto_vertical or self.cfg.tilt_deg >= 0:
            return
        self._aspect = size[1] / size[0]
        for person in people:
            if t - self._last_sample.get(person.track_id, -math.inf) < SAMPLE_EVERY_S:
                continue
            lines = _body_lines(person, min_conf)
            if not lines:
                continue
            self._last_sample[person.track_id] = t
            for base, direction, weight in lines:
                p = self._normalise(base, size)
                d = direction / np.linalg.norm(direction)
                self._samples.append((p[0], p[1], d[0], d[1], weight))
                self._fresh += 1
        if len(self._last_sample) > 500:
            self._last_sample = {k: v for k, v in self._last_sample.items() if t - v < 60.0}
        if self._fresh >= 20 and len(self._samples) >= MIN_SAMPLES and t - self._last_fit >= REFIT_EVERY_S:
            self._last_fit, self._fresh = t, 0
            self._fit()

    def _fit(self) -> None:
        s = np.array(self._samples)
        points, dirs, base_weight = s[:, :2], s[:, 2:4], s[:, 4]
        # People in one or two spots cannot tell a tilted camera from leaning people:
        # the body lines must come from places spread across the picture.
        columns = np.clip(((points[:, 0] + 0.5) * 6).astype(int), 0, 5)
        rows = np.clip(((points[:, 1] / max(self._aspect, 1e-6) + 0.5) * 4).astype(int), 0, 3)
        if len(set(columns)) < 3 or len(set(zip(columns, rows))) < MIN_PLACES:
            return
        normals = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1)
        lines = np.concatenate([normals, -(normals * points).sum(1, keepdims=True)], axis=1)

        # The point closest to all body lines, found again and again while
        # giving less say to lines that miss it (people who are bending).
        weight = base_weight.copy()
        vp, off = None, None
        for _ in range(12):
            scatter = (lines * weight[:, None]).T @ lines
            vp = np.linalg.eigh(scatter)[1][:, 0]
            away = self._away(vp, points)
            off = np.degrees(np.arccos(np.abs((away * dirs).sum(1)).clip(0.0, 1.0)))
            weight = base_weight / (1.0 + (off / 10.0) ** 2)

        agree = off < 20.0
        if agree.mean() < 0.6 or np.median(off[agree]) > 10.0:
            return  # the body lines do not meet in one point well enough
        sign = 1.0 if (((self._away(vp, points) * dirs).sum(1)) * weight).sum() >= 0 else -1.0
        # Only two readings are believed: a level camera, or one that looks down. Then "up" at
        # the centre of the picture points up it, and the meeting point lies below the picture.
        # Anything else is more likely people leaning the same way than a camera on its side.
        up_at_centre = sign * self._away(vp, np.zeros((1, 2)))[0]
        if math.degrees(math.atan2(abs(up_at_centre[0]), -up_at_centre[1])) > MAX_ROLL_DEG:
            return
        finite = abs(vp[2]) > 1e-9 * np.linalg.norm(vp[:2])
        if finite and np.linalg.norm(vp[:2] / vp[2]) < FAR_AWAY:
            if sign < 0 or vp[1] / vp[2] < 0.5 * self._aspect:
                return

        gravity = np.array([vp[0], vp[1], vp[2] * self._focal])
        self._vp, self._sign = vp, sign
        self._gravity = gravity / np.linalg.norm(gravity)
        self.pitch_deg = math.degrees(math.asin(min(1.0, abs(self._gravity[2]))))
        self.learned = True

    @staticmethod
    def _away(vp: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Unit vectors pointing from the vanishing point toward each point."""
        if abs(vp[2]) > 1e-9 * np.linalg.norm(vp[:2]):
            return _unit_rows(points - vp[:2] / vp[2])
        return _unit_rows(np.tile(vp[:2], (len(points), 1)))  # vanishing point at infinity: one direction

    @staticmethod
    def _normalise(point, size: tuple[int, int]) -> np.ndarray:
        w, h = size
        return np.array([(point[0] - w / 2.0) / w, (point[1] - h / 2.0) / w])

    # --- use -----------------------------------------------------------------

    def frame_at(self, point, size: tuple[int, int]) -> tuple[np.ndarray, float]:
        """(unit vector pointing up, squash factor of a standing person) at this place in the picture."""
        if self._vp is None:
            return UP, 1.0
        p = self._normalise(point, size)
        up = self._sign * self._away(self._vp, p[None])[0]
        # Length in the picture of a short vertical line, next to a level sideways line of the
        # same real length at the same place. 1 for a level camera, less the steeper it looks down.
        g = self._gravity
        squash = float(np.linalg.norm(g[:2] - (p / self._focal) * g[2]))
        return up, min(1.0, max(MIN_SQUASH, squash))

    def rectify(self, person: Person, size: tuple[int, int], min_conf: float) -> tuple[Person, float]:
        """The same skeleton as a level camera would show it, and the squash factor that was undone."""
        if self._vp is None:
            return person, 1.0
        _, hip = trunk_points(person, min_conf)
        if hip is None:
            x1, y1, x2, y2 = person.box
            hip = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        up, squash = self.frame_at(hip, size)
        right = np.array([-up[1], up[0]])
        rel = person.xy.astype(float) - hip
        xy = np.stack([hip[0] + rel @ right, hip[1] - (rel @ up) / squash], axis=1)
        return Person(person.track_id, xy, person.conf, person.box), squash


def _body_lines(person: Person, min_conf: float) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """A line that is vertical when the person stands upright: (a point on it, direction, weight)."""
    # A straight leg is the better witness: it stays vertical when the trunk bends.
    leg = leg_points(person, min_conf)
    if leg is not None:
        _, hip_s, ankle, straight = leg
        if straight and np.linalg.norm(hip_s - ankle) > 10.0:
            return [(ankle, hip_s - ankle, 2.0)]
    shoulder, hip = trunk_points(person, min_conf)
    if shoulder is not None and hip is not None and np.linalg.norm(shoulder - hip) > 10.0:
        return [(hip, shoulder - hip, 1.0)]
    return []
