"""Tracks how long each worker holds each posture and flags the bad ones that last."""

from __future__ import annotations

import itertools
from collections import deque

from .angles import Angles
from .config import AngleConfig, TimingConfig
from .reba import knee_bend

LEVELS = ("safe", "caution", "bad")
REGIONS = ("back", "neck", "shoulders", "knees")
# Vibration code of the ankle pod: 1 pulse = arms and shoulders, 2 = trunk and back, 3 = legs and feet.
PULSES = {"shoulders": 1, "back": 2, "neck": 2, "knees": 3}
# When two regions are equally bad, the first in this list names the worker's posture.
PRIORITY = ("back", "shoulders", "knees", "neck")
# Flags are numbered 1, 2, 3... for as long as the app runs, across Stop and Start, so a client
# that remembers the last number it handled can ask for newer flags only (GET /flags?after=N).
_FLAG_NUMBERS = itertools.count(1)


def describe(angles: Angles, cfg: AngleConfig) -> dict[str, tuple[str, float | None]]:
    """A short phrase and the main angle for each region, for people to read."""
    out: dict[str, tuple[str, float | None]] = {}

    flex, side = angles.trunk_flexion, angles.trunk_side_bend
    if flex is None:
        out["back"] = ("Not visible", None)
    elif flex > 20:
        out["back"] = (f"Bent forward {flex:.0f}°", flex)
    elif flex < -cfg.upright_tol_deg:
        out["back"] = (f"Leaning back {-flex:.0f}°", flex)
    elif side is not None and side > cfg.side_bend_deg:
        out["back"] = (f"Bent sideways {side:.0f}°", side)
    else:
        out["back"] = ("Upright", flex)

    flex, side = angles.neck_flexion, angles.neck_side_bend
    if flex is None and side is None:
        out["neck"] = ("Not visible", None)
    elif flex is not None and flex > 20:
        out["neck"] = (f"Neck bent {flex:.0f}°", flex)
    elif flex is not None and flex < -cfg.neck_extension_deg:
        out["neck"] = (f"Neck tilted back {-flex:.0f}°", flex)
    elif side is not None and side > cfg.neck_side_bend_deg:
        out["neck"] = (f"Neck tilted sideways {side:.0f}°", side)
    else:
        out["neck"] = ("Neutral", flex if flex is not None else side)

    lifts = [v for v in angles.upper_arm.values() if v is not None]
    if not lifts:
        out["shoulders"] = ("Not visible", None)
    else:
        top = max(lifts)
        if top > 90:
            out["shoulders"] = (f"Arm above shoulder {top:.0f}°", top)
        elif top > 45:
            out["shoulders"] = (f"Arm raised {top:.0f}°", top)
        elif top > 20:
            out["shoulders"] = (f"Arm lifted {top:.0f}°", top)
        elif min(lifts) < -20:
            out["shoulders"] = (f"Arm behind body {-min(lifts):.0f}°", min(lifts))
        else:
            out["shoulders"] = ("Arms down", top)

    bend = knee_bend(angles.knee)
    if bend is None:
        out["knees"] = ("Not visible", None)
    else:
        if bend > 60:
            out["knees"] = (f"Deep knee bend {bend:.0f}°", bend)
        elif bend >= 30:
            out["knees"] = (f"Knees bent {bend:.0f}°", bend)
        else:
            out["knees"] = ("Legs straight", bend)
    return out


class RegionTimer:
    """Level of one body region over time: how long it has been held and when to flag it."""

    def __init__(self, cfg: TimingConfig):
        self.cfg = cfg
        self.level: int | None = None  # the confirmed level; None while the region is not visible
        self.since: float | None = None  # when the confirmed level began
        self.totals = [0.0, 0.0, 0.0]  # seconds spent safe, caution, bad
        self.flag_count = 0
        self._candidate: int | None = None
        self._candidate_since = 0.0
        self._last_seen: float | None = None
        self._last_t: float | None = None
        self._last_flag: float | None = None

    def update(self, t: float, observed: int | None) -> bool:
        """Feed the level seen at time t. Returns True when a flag fires."""
        if self._last_t is not None and self.level is not None:
            dt = t - self._last_t
            if 0 < dt <= self.cfg.lost_grace_s:
                self.totals[self.level] += dt
        self._last_t = t

        if observed is None:
            # Not visible: hold the level through a short gap, then stop the clock.
            if self._last_seen is None or t - self._last_seen > self.cfg.lost_grace_s:
                self.level, self.since, self._candidate = None, None, None
            return False
        self._last_seen = t

        if self.level is None:
            self.level, self.since, self._candidate = observed, t, None
        elif observed == self.level:
            self._candidate = None
        elif observed != self._candidate:
            self._candidate, self._candidate_since = observed, t
        elif t - self._candidate_since >= self.cfg.debounce_s:
            # The new level has lasted: it counts from the moment it first appeared.
            self.level, self.since, self._candidate = observed, self._candidate_since, None
        return self._flag_due(t)

    def held(self, t: float) -> float:
        return 0.0 if self.since is None else max(0.0, t - self.since)

    def over_hold(self, t: float) -> bool:
        """True while a caution or bad posture has been held past its limit."""
        if not self.level:
            return False
        hold = self.cfg.bad_hold_s if self.level == 2 else self.cfg.caution_hold_s
        return self.held(t) >= hold

    def _flag_due(self, t: float) -> bool:
        if not self.over_hold(t):
            return False
        if self._last_flag is not None and t - self._last_flag < self.cfg.cooldown_s:
            return False
        self._last_flag = t
        self.flag_count += 1
        return True


class PostureMonitor:
    """Timers for every tracked worker, and the list of flags raised."""

    def __init__(self, timing: TimingConfig):
        self.timing = timing
        self.workers: dict[int, dict[str, RegionTimer]] = {}
        self.last_seen: dict[int, float] = {}
        self.flags: deque[dict] = deque(maxlen=200)
        # The words for the posture being timed. A level changes only after it has lasted,
        # so the words must not follow every frame either, or text and colour would disagree.
        self._labels: dict[tuple[int, str], tuple[str, float | None]] = {}

    def is_static(self, worker_id: int, t: float) -> bool:
        """True when the worker has held a caution or bad posture for over a minute (REBA activity rule)."""
        timers = self.workers.get(worker_id)
        if not timers:
            return False
        return any(tm.level and tm.held(t) >= self.timing.static_s for tm in timers.values())

    def update(self, t: float, worker_id: int, levels: dict[str, int | None],
               labels: dict[str, tuple[str, float | None]], clock: str = "") -> list[dict]:
        """Feed one worker's region levels at time t. Returns the flags that fired."""
        timers = self.workers.setdefault(worker_id, {r: RegionTimer(self.timing) for r in REGIONS})
        self.last_seen[worker_id] = t
        fired = []
        for region in REGIONS:
            timer = timers[region]
            flag_due = timer.update(t, levels.get(region))
            if region in labels and timer.level is not None and levels.get(region) == timer.level:
                self._labels[(worker_id, region)] = labels[region]
            if not flag_due:
                continue
            label, angle = self._labels.get((worker_id, region), labels.get(region, (region, None)))
            held = timer.held(t)
            flag = {
                "seq": next(_FLAG_NUMBERS),
                "t": round(t, 1),
                "clock": clock,
                "worker": worker_id,
                "region": region,
                "level": LEVELS[timer.level],
                "angle": None if angle is None else round(angle),
                "held_s": round(held, 1),
                "pulses": PULSES[region],
                "message": f"Worker {worker_id}: {label[0].lower()}{label[1:]} for {held:.0f} s",
            }
            self.flags.append(flag)
            fired.append(flag)
        return fired

    def view(self, worker_id: int, t: float) -> dict[str, dict]:
        """Current level, words, time held and flag state per region, ready for display."""
        out = {}
        for region, timer in self.workers.get(worker_id, {}).items():
            label, angle = ("Not visible", None)
            if timer.level is not None:
                label, angle = self._labels.get((worker_id, region), ("", None))
            out[region] = {
                "level": None if timer.level is None else LEVELS[timer.level],
                "label": label,
                "angle": None if angle is None else round(angle),
                "held_s": round(timer.held(t), 1),
                "flagged": timer.over_hold(t),
            }
        return out

    def prune(self, t: float) -> list[int]:
        """Forget workers that have left the picture. Returns their ids."""
        gone = [w for w, seen in self.last_seen.items() if t - seen > self.timing.track_timeout_s]
        for w in gone:
            del self.workers[w]
            del self.last_seen[w]
            for region in REGIONS:
                self._labels.pop((w, region), None)
        return gone

    def clear_workers(self) -> None:
        """Drop all timers but keep the flag history (used when the video restarts)."""
        self.workers.clear()
        self.last_seen.clear()
        self._labels.clear()
