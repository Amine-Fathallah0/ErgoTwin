"""Draws each worker's skeleton coloured by posture level, with a label."""

from __future__ import annotations

import cv2
import numpy as np

from .skeleton import KEYPOINTS, Person

# BGR colours, matching the web page.
COLOURS = {"safe": (113, 191, 47), "caution": (36, 165, 245), "bad": (68, 68, 239), None: (150, 150, 150)}
JOINT = (245, 245, 245)
FONT = cv2.FONT_HERSHEY_SIMPLEX
CROWD = 5  # with more people than this in view, labels shrink to what needs attention

# Each bone takes the colour of the body region it belongs to.
BONES = (
    ("l_shoulder", "r_shoulder", "back"), ("l_hip", "r_hip", "back"),
    ("l_shoulder", "l_hip", "back"), ("r_shoulder", "r_hip", "back"),
    ("l_shoulder", "l_elbow", "shoulders"), ("l_elbow", "l_wrist", "shoulders"),
    ("r_shoulder", "r_elbow", "shoulders"), ("r_elbow", "r_wrist", "shoulders"),
    ("l_hip", "l_knee", "knees"), ("l_knee", "l_ankle", "knees"),
    ("r_hip", "r_knee", "knees"), ("r_knee", "r_ankle", "knees"),
)


def _clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def _px(point) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def _mid(a, b):
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a if a is not None else b


def _size(person: Person) -> float:
    x1, y1, x2, y2 = person.box
    return max(x2 - x1, y2 - y1)


def render(frame: np.ndarray, people: list[Person], views: dict[int, dict], min_conf: float,
           privacy: bool) -> np.ndarray:
    """The frame with skeletons and labels drawn on it. In privacy mode the video itself is left out."""
    canvas = np.zeros_like(frame) if privacy else frame
    shown = [(person, views[person.track_id]) for person in people if person.track_id in views]
    for person, view in shown:
        # Line width follows the person's size, so small distant workers are not buried in paint.
        thick = max(1, min(4, round(_size(person) / 70)))
        _skeleton(canvas, person, view["regions"], min_conf, thick)
    for person, view in shown:  # labels last, so no skeleton is drawn over them
        _label(canvas, person, view, compact=len(shown) > CROWD)
    return canvas


def _skeleton(canvas, person: Person, regions: dict, min_conf: float, thick: int) -> None:
    def colour(region: str):
        return COLOURS.get(regions.get(region, {}).get("level"), COLOURS[None])

    for a, b, region in BONES:
        pa, pb = person.pt(a, min_conf), person.pt(b, min_conf)
        if pa is not None and pb is not None:
            cv2.line(canvas, _px(pa), _px(pb), colour(region), thick, cv2.LINE_AA)

    shoulder = _mid(person.pt("l_shoulder", min_conf), person.pt("r_shoulder", min_conf))
    head = _mid(person.pt("l_ear", min_conf), person.pt("r_ear", min_conf))
    if head is None:
        head = person.pt("nose", min_conf)
    if shoulder is not None and head is not None:
        cv2.line(canvas, _px(shoulder), _px(head), colour("neck"), thick, cv2.LINE_AA)
        cv2.circle(canvas, _px(head), thick + 1, colour("neck"), -1, cv2.LINE_AA)

    for name in KEYPOINTS[5:]:  # body joints; the face points are left out
        point = person.pt(name, min_conf)
        if point is not None:
            cv2.circle(canvas, _px(point), max(2, thick), JOINT, -1, cv2.LINE_AA)


def _label(canvas, person: Person, view: dict, compact: bool) -> None:
    x1, y1, x2, y2 = (int(v) for v in person.box)
    level = view["level"]
    alert = level in ("caution", "bad")
    colour = COLOURS.get(level, COLOURS[None])
    if view["flagged"]:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOURS["bad"], 2)

    # In a crowd, only workers who need attention get words; the rest get their number.
    text = f"#{view['id']}"
    if alert or not compact:
        text += f" {view['posture']}"
    if alert:
        text += f"  {_clock(view['held_s'])}"
    text = text.replace("°", " deg")  # OpenCV's built-in font has no degree sign

    scale = max(0.42, min(0.8, _size(person) / 260))
    weight = 1 if scale < 0.6 else 2
    (tw, th), base = cv2.getTextSize(text, FONT, scale, weight)
    x = max(0, min(x1, canvas.shape[1] - tw - 6))
    y = max(th + 6, y1 - 4)
    cv2.rectangle(canvas, (x, y - th - 6), (x + tw + 6, y + base - 2), colour, -1)
    cv2.putText(canvas, text, (x + 3, y - 3), FONT, scale, (20, 20, 20), weight, cv2.LINE_AA)
