"""From tracked skeletons to posture state. This part does not care where the frames come from."""

from __future__ import annotations

from .angles import AngleSmoother, TorsoReference, measure
from .camera import LEVEL_SQUASH, VerticalField
from .config import Config
from .posture import PRIORITY, PULSES, REGIONS, PostureMonitor, describe
from .reba import RebaResult, region_levels, score
from .skeleton import Person

_RANK = {None: -1, "safe": 0, "caution": 1, "bad": 2}


def worker_view(worker_id: int, reba: RebaResult, timers: dict, view: str) -> dict:
    """Everything about one worker that the page, the dashboard and the pod need."""
    regions = {region: {**timers[region], "pulses": PULSES[region]} for region in REGIONS}
    worst = max(PRIORITY, key=lambda r: _RANK[regions[r]["level"]])  # ties go to the first in PRIORITY
    level = regions[worst]["level"]
    alert = level in ("caution", "bad")
    return {
        "id": worker_id,
        "level": level,
        "posture": regions[worst]["label"] if alert else "Neutral posture" if level else "Not visible",
        "held_s": regions[worst]["held_s"] if alert else 0.0,
        "flagged": any(r["flagged"] for r in regions.values()),
        "view": view,
        "reba": {
            "score": reba.score, "risk": reba.risk, "trunk": reba.trunk, "neck": reba.neck, "legs": reba.legs,
            "upper_arm": reba.upper_arm, "lower_arm": reba.lower_arm, "wrist": reba.wrist,
            "activity": reba.activity,
            # False when a region is hidden: the score then counts that region as neutral.
            "complete": all(r["level"] is not None for r in regions.values()),
        },
        "regions": regions,
    }


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.monitor = PostureMonitor(cfg.timing)
        self.camera = VerticalField(cfg.camera)
        # The tracker's own ids only ever grow, and a busy scene burns through them. Workers are
        # shown under small numbers instead, and a number is free again once its worker has left.
        self._numbers: dict[int, int] = {}
        self._first_seen: dict[int, float] = {}
        self._references: dict[int, TorsoReference] = {}
        self._smoothers: dict[int, AngleSmoother] = {}

    @property
    def flags(self):
        return self.monitor.flags

    def process(self, t: float, people: list[Person], size: tuple[int, int], clock: str = "") -> dict[int, dict]:
        """One frame: the people found at time t, in a picture of (width, height). Returns a view per worker."""
        cfg = self.cfg
        min_conf = cfg.pose.keypoint_conf
        self.camera.observe(t, people, size, min_conf)

        views: dict[int, dict] = {}
        for person in people:
            wid = person.track_id
            first_seen = self._first_seen.setdefault(wid, t)
            upright, squash = self.camera.rectify(person, size, min_conf)
            tilted = squash < LEVEL_SQUASH
            torso_ref = None
            if not tilted:  # reading depth from the legs assumes a level camera
                torso_ref = self._references.setdefault(wid, TorsoReference()).update(t, upright, min_conf)
            if t - first_seen < cfg.timing.min_track_s:
                continue  # may be a false detection that is gone in a moment
            angles = measure(upright, cfg.angles, min_conf, torso_ref, tilted)
            if angles.trunk_flexion is None:
                continue  # too little of this person is visible to say anything about posture
            angles = self._smoothers.setdefault(wid, AngleSmoother(cfg.angles.smoothing)).apply(angles)
            number = self._numbers.get(wid)
            if number is None:
                number = self._numbers[wid] = self._free_number()
            reba = score(angles, cfg.station, cfg.angles, static=self.monitor.is_static(number, t))
            levels = region_levels(reba, angles, cfg.levels)
            self.monitor.update(t, number, levels, describe(angles, cfg.angles), clock=clock)
            views[wid] = worker_view(number, reba, self.monitor.view(number, t), angles.view)

        gone = set(self.monitor.prune(t))
        self._numbers = {wid: n for wid, n in self._numbers.items() if n not in gone}
        in_view = {person.track_id for person in people}
        for store in (self._first_seen, self._references, self._smoothers):
            for wid in [w for w in store if w not in in_view and w not in self._numbers]:
                del store[wid]
        return views

    def _free_number(self) -> int:
        taken = set(self._numbers.values())
        number = 1
        while number in taken:
            number += 1
        return number
