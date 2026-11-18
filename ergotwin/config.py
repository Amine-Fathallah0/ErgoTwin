"""Settings for the camera module. Defaults live here; config.toml overrides them."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class PoseConfig:
    model: str = "yolo26m-pose.pt"  # weights file kept in models/ (downloaded on first run)
    device: str = "auto"  # "auto" = GPU when there is one, else CPU; or "cpu", "0"
    imgsz: int = 1280  # size the model works at; 640 is faster but misses small, distant people
    conf: float = 0.35  # minimum confidence to start following a detected person
    keypoint_conf: float = 0.5  # minimum confidence to use a joint
    half: bool = True  # half precision on the GPU
    max_people: int = 40
    track_memory_frames: int = 45  # a hidden worker keeps their number for this many frames
    max_width: int = 1280  # wider frames are scaled down before processing


@dataclass
class StationConfig:
    """What the camera cannot see. Set these per workstation."""

    seated: bool = False  # seated work: bent knees are not penalised
    load_kg: float = 0.0  # weight handled
    shock: bool = False  # shock or rapid build-up of force
    coupling: int = 0  # grip: 0 good, 1 fair, 2 poor, 3 unacceptable
    wrist_score: int = 1  # 1 neutral, 2 bent over 15 degrees, 3 bent and twisted
    arm_supported: bool = False  # arm supported, or the worker is leaning
    repeated_actions: bool = False  # small actions repeated more than 4 times a minute
    rapid_changes: bool = False  # rapid large posture changes or an unstable base


@dataclass
class AngleConfig:
    upright_tol_deg: float = 5.0  # a trunk within this of vertical counts as upright
    side_bend_deg: float = 15.0  # trunk bent sideways beyond this adds 1 to the REBA score
    neck_side_bend_deg: float = 25.0  # same for the neck, which is short and so noisier
    neck_neutral_deg: float = 10.0  # the ear sits this far ahead of the shoulder in a neutral neck
    neck_extension_deg: float = 15.0  # neck tilted back beyond this counts as extension
    abduction_deg: float = 45.0  # arm lifted sideways beyond this counts as abducted
    front_view_ratio: float = 0.55  # shoulder width / torso length above this: facing the camera
    side_view_ratio: float = 0.30  # below this: seen from the side
    foreshortening: bool = True  # read a short-looking torso as a bend toward the camera
    foreshortening_min_deg: float = 25.0  # ignore smaller such bends: they are within the noise
    smoothing: float = 0.5  # weight of the newest frame: 1 = no smoothing, lower = smoother
    min_person_px: float = 50.0  # people smaller than this in the picture are not measured
    min_neck_px: float = 25.0  # the neck is measured only when it is at least this long in the picture


@dataclass
class CameraConfig:
    auto_vertical: bool = True  # learn which way is "up" in the picture from the workers themselves
    tilt_deg: float = -1.0  # how far the camera looks down, if you know it (0 = level); -1 = learn it
    hfov_deg: float = 60.0  # the lens's horizontal field of view; matters only for cameras that look down
    tile: str = "whole"  # a grid recording: "3x3:5" analyses tile 5 of a 3 x 3 grid, "3x3:all" every tile


@dataclass
class GridConfig:
    """How every tile of a grid recording is analysed at once ("3x3:all"). One tile alone uses [pose]."""

    model: str = "yolo26s-pose.pt"  # nine tiles per frame: the small model keeps it near 7 frames a second
    tile_width: int = 1088  # each tile is enlarged to this width before the model looks at it
    conf: float = 0.25  # lower than [pose]: people in a grid are small and the model is less sure of them


@dataclass
class TimingConfig:
    debounce_s: float = 0.6  # a new posture must last this long before it counts
    lost_grace_s: float = 1.0  # keep the timer running through detection gaps this short
    bad_hold_s: float = 5.0  # flag a bad posture held this long
    caution_hold_s: float = 60.0  # flag a caution posture held this long
    cooldown_s: float = 20.0  # minimum time between two flags for the same region
    static_s: float = 60.0  # REBA adds 1 when a posture is held this long
    track_timeout_s: float = 3.0  # forget a worker not seen for this long
    min_track_s: float = 1.0  # show a worker only once followed this long: brief false detections stay out


def _default_levels() -> dict[str, list[int]]:
    # REBA part score at which a region turns [caution, bad].
    return {"back": [3, 4], "neck": [2, 3], "shoulders": [3, 4], "knees": [2, 3]}


@dataclass
class Config:
    pose: PoseConfig = field(default_factory=PoseConfig)
    station: StationConfig = field(default_factory=StationConfig)
    angles: AngleConfig = field(default_factory=AngleConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    levels: dict[str, list[int]] = field(default_factory=_default_levels)


def load_config(path: Path | str | None = None) -> Config:
    """Defaults, overridden by the TOML file if it exists. Unknown names are errors, to catch typos."""
    cfg = Config()
    if path is None or not Path(path).exists():
        return cfg
    with open(path, "rb") as f:
        data = tomllib.load(f)
    for section, values in data.items():
        if section == "levels":
            for region, cuts in values.items():
                if region not in cfg.levels:
                    raise ValueError(f"Unknown region levels.{region} in {path}")
                cfg.levels[region] = [int(cuts[0]), int(cuts[1])]
            continue
        if section not in {f.name for f in fields(cfg)}:
            raise ValueError(f"Unknown section [{section}] in {path}")
        target = getattr(cfg, section)
        known = {f.name for f in fields(target)}
        for key, value in values.items():
            if key not in known:
                raise ValueError(f"Unknown setting {section}.{key} in {path}")
            setattr(target, key, value)
    return cfg
