"""REBA (Rapid Entire Body Assessment) scoring, after Hignett and McAtamney (2000).

The camera supplies the joint angles. Everything it cannot see (load, grip,
wrist posture, seated work) comes from the station settings.
"""

from __future__ import annotations

from dataclasses import dataclass

from .angles import SIDES, Angles
from .config import AngleConfig, StationConfig

# Table A: [trunk - 1][neck - 1][legs - 1]
TABLE_A = [
    [[1, 2, 3, 4], [1, 2, 3, 4], [3, 3, 5, 6]],
    [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
    [[2, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
    [[3, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
    [[4, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],
]

# Table B: [upper arm - 1][lower arm - 1][wrist - 1]
TABLE_B = [
    [[1, 2, 2], [1, 2, 3]],
    [[1, 2, 3], [2, 3, 4]],
    [[3, 4, 5], [4, 5, 5]],
    [[4, 5, 5], [5, 6, 7]],
    [[6, 7, 8], [7, 8, 8]],
    [[7, 8, 8], [8, 9, 9]],
]

# Table C: [score A - 1][score B - 1]
TABLE_C = [
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
]


@dataclass
class RebaResult:
    trunk: int
    neck: int
    legs: int
    upper_arm: int
    lower_arm: int
    wrist: int
    score_a: int  # Table A plus load
    score_b: int  # Table B plus coupling
    score_c: int
    activity: int
    score: int  # final REBA score, 1 to 15
    risk: str


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def trunk_score(flexion: float | None, side_bend: float | None, cfg: AngleConfig) -> int:
    if flexion is None:
        return 1
    if abs(flexion) <= cfg.upright_tol_deg:
        s = 1
    elif flexion < 0:  # leaning back
        s = 2 if flexion >= -20 else 3
    elif flexion <= 20:
        s = 2
    elif flexion <= 60:
        s = 3
    else:
        s = 4
    if side_bend is not None and side_bend > cfg.side_bend_deg:
        s += 1
    return s


def neck_score(flexion: float | None, side_bend: float | None, cfg: AngleConfig) -> int:
    s = 1
    if flexion is not None and (flexion > 20 or flexion < -cfg.neck_extension_deg):
        s = 2
    if side_bend is not None and side_bend > cfg.neck_side_bend_deg:
        s += 1
    return s


def knee_bend(knee: dict) -> float | None:
    """The knee bend that counts for the legs: the straighter knee decides.

    Crouching bends both knees. Walking bends one at a time, which REBA does
    not penalise. With one leg hidden, the visible knee is used.
    """
    bends = [k for k in knee.values() if k is not None]
    return min(bends) if bends else None


def legs_score(knee: dict, seated: bool) -> int:
    # Both feet are assumed to be on the ground: one-leg stance cannot be read reliably.
    s = 1
    bend = knee_bend(knee)
    if bend is not None and not seated:
        if bend > 60:
            s += 2
        elif bend >= 30:
            s += 1
    return s


def upper_arm_score(lift: float | None, abducted: bool, supported: bool) -> int:
    if lift is None:
        return 1
    if lift < -20:  # arm behind the body
        s = 2
    elif lift <= 20:
        s = 1
    elif lift <= 45:
        s = 2
    elif lift <= 90:
        s = 3
    else:
        s = 4
    if abducted:
        s += 1
    if supported:
        s -= 1
    return _clamp(s, 1, 6)


def lower_arm_score(elbow: float | None) -> int:
    if elbow is None:
        return 1
    return 1 if 60 <= elbow <= 100 else 2


def load_score(station: StationConfig) -> int:
    s = 0 if station.load_kg < 5 else 1 if station.load_kg <= 10 else 2
    return s + (1 if station.shock else 0)


def risk_level(score: int) -> str:
    if score <= 1:
        return "negligible"
    if score <= 3:
        return "low"
    if score <= 7:
        return "medium"
    if score <= 10:
        return "high"
    return "very high"


def score(angles: Angles, station: StationConfig, cfg: AngleConfig, static: bool = False) -> RebaResult:
    """REBA score of one posture. `static` is True when a posture has been held over a minute."""
    trunk = _clamp(trunk_score(angles.trunk_flexion, angles.trunk_side_bend, cfg), 1, 5)
    neck = _clamp(neck_score(angles.neck_flexion, angles.neck_side_bend, cfg), 1, 3)
    legs = _clamp(legs_score(angles.knee, station.seated), 1, 4)

    arms = {s: upper_arm_score(angles.upper_arm[s], angles.arm_abducted[s], station.arm_supported) for s in SIDES}
    worst = max(SIDES, key=lambda s: arms[s])  # REBA scores the more loaded arm
    other = "r" if worst == "l" else "l"
    upper = arms[worst]
    elbow = angles.elbow[worst] if angles.elbow[worst] is not None else angles.elbow[other]
    lower = lower_arm_score(elbow)
    wrist = _clamp(station.wrist_score, 1, 3)

    score_a = TABLE_A[trunk - 1][neck - 1][legs - 1] + load_score(station)
    score_b = TABLE_B[upper - 1][lower - 1][wrist - 1] + _clamp(station.coupling, 0, 3)
    score_c = TABLE_C[_clamp(score_a, 1, 12) - 1][_clamp(score_b, 1, 12) - 1]
    activity = int(static) + int(station.repeated_actions) + int(station.rapid_changes)
    total = score_c + activity
    return RebaResult(trunk, neck, legs, upper, lower, wrist, score_a, score_b, score_c, activity, total,
                      risk_level(total))


def _level(part_score: int, cuts: list[int]) -> int:
    return 2 if part_score >= cuts[1] else 1 if part_score >= cuts[0] else 0


def region_levels(reba: RebaResult, angles: Angles, cuts: dict[str, list[int]]) -> dict[str, int | None]:
    """Level per body region: 0 safe, 1 caution, 2 bad, None when the region is not visible."""
    seen_neck = angles.neck_flexion is not None or angles.neck_side_bend is not None
    seen_arm = any(v is not None for v in angles.upper_arm.values())
    seen_knee = any(v is not None for v in angles.knee.values())
    return {
        "back": _level(reba.trunk, cuts["back"]) if angles.trunk_flexion is not None else None,
        "neck": _level(reba.neck, cuts["neck"]) if seen_neck else None,
        "shoulders": _level(reba.upper_arm, cuts["shoulders"]) if seen_arm else None,
        "knees": _level(reba.legs, cuts["knees"]) if seen_knee else None,
    }
