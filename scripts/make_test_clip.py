"""Builds a synthetic test clip from a sample photo: people upright, then tilted as if bent over.

It exercises the whole chain (pose model, angles, timers, flags, web page) when
no real footage is at hand. It is not a demo clip.

Usage:  .venv\\Scripts\\python scripts\\make_test_clip.py   ->  videos/synthetic_test.mp4
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils import ASSETS

FPS = 15
SIDE = 960
# (seconds, tilt in degrees): upright, tilted long enough to be flagged, upright again.
SEGMENTS = ((8, 0.0), (12, 70.0), (6, 0.0))
RAMP_S = 1.0  # time taken to move from one tilt to the next


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "videos" / "synthetic_test.mp4"
    out.parent.mkdir(exist_ok=True)
    photo = cv2.imread(str(ASSETS / "bus.jpg"))
    h, w = photo.shape[:2]

    # Centre the photo on a square big enough that turning it never cuts anyone off.
    side = math.ceil(math.hypot(w, h))
    canvas = np.full((side, side, 3), 40, np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = photo

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (SIDE, SIDE))
    previous = SEGMENTS[0][1]
    for seconds, tilt in SEGMENTS:
        for i in range(int(seconds * FPS)):
            blend = min(1.0, i / (RAMP_S * FPS))
            angle = previous + (tilt - previous) * blend
            turn = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
            frame = cv2.warpAffine(canvas, turn, (side, side), borderValue=(40, 40, 40))
            writer.write(cv2.resize(frame, (SIDE, SIDE), interpolation=cv2.INTER_AREA))
        previous = tilt
    writer.release()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
