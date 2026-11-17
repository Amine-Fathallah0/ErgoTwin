"""One detected person as named keypoints, independent of the pose model used."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The 17 COCO keypoints. Left and right are the worker's own left and right.
KEYPOINTS = (
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
)
INDEX = {name: i for i, name in enumerate(KEYPOINTS)}


@dataclass
class Person:
    track_id: int
    xy: np.ndarray  # (17, 2) pixel coordinates: x to the right, y down
    conf: np.ndarray  # (17,) keypoint confidence, 0 to 1
    box: tuple[float, float, float, float]  # x1, y1, x2, y2

    def pt(self, name: str, min_conf: float) -> np.ndarray | None:
        """The keypoint, or None when the model is not confident about it."""
        i = INDEX[name]
        if self.conf[i] < min_conf:
            return None
        return self.xy[i].astype(float)
