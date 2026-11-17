"""Pose backend: finds and tracks people with a YOLO pose model, on the GPU when there is one.

To use another pose model, write a class with the same three members
(`device_name`, `infer`, `reset`) that returns `Person` objects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import PoseConfig
from .skeleton import Person


TRACK_LOW = 0.1  # a detection this weak may continue a track, never start one


def _tracker_settings(cfg: PoseConfig) -> str:
    """ByteTrack settings file, built from config.toml."""
    return (
        "# Written by ergotwin/pose.py on every start. Change config.toml, not this file.\n"
        "tracker_type: bytetrack\n"
        f"track_high_thresh: {cfg.conf}\n"
        f"track_low_thresh: {TRACK_LOW}\n"
        f"new_track_thresh: {cfg.conf}\n"
        f"track_buffer: {cfg.track_memory_frames}\n"
        "match_thresh: 0.8\n"
        "fuse_score: True\n"
    )


class YoloPose:
    def __init__(self, cfg: PoseConfig, models_dir: Path):
        # Imported here so the scoring code and its tests work without PyTorch installed.
        import torch
        from ultralytics import YOLO

        # ErgoTwin promises that nothing about the workers leaves the laptop, so the library's
        # anonymous usage statistics are switched off, for this app only.
        try:
            from ultralytics.utils import events as analytics

            analytics.events.enabled = False
        except (ImportError, AttributeError):  # a library version without them
            pass

        self.cfg = cfg
        on_gpu = cfg.device != "cpu" and torch.cuda.is_available()
        self.device = (0 if cfg.device == "auto" else cfg.device) if on_gpu else "cpu"
        self.half = cfg.half and on_gpu
        self.device_name = torch.cuda.get_device_name(0) if on_gpu else "CPU"
        models_dir.mkdir(parents=True, exist_ok=True)
        # A weights file that is missing is downloaded to this path on first use.
        self.model = YOLO(str(models_dir / cfg.model))
        self._tracker_file = models_dir / "tracker.yaml"
        self._tracker_file.write_text(_tracker_settings(cfg), encoding="utf-8")

    def infer(self, frame: np.ndarray) -> list[Person]:
        """People in one BGR frame, each with a track id that stays the same across frames."""
        precision = {"quantize": 16} if self.half else {}  # 16-bit floats on the GPU
        # The model's own cut-off is set low so the tracker also sees weak detections: a worker
        # half hidden for a moment keeps their number. New workers still need cfg.conf.
        result = self.model.track(
            frame, persist=True, tracker=str(self._tracker_file), device=self.device, imgsz=self.cfg.imgsz,
            conf=TRACK_LOW, max_det=self.cfg.max_people, verbose=False, **precision,
        )[0]
        if result.boxes is None or result.boxes.id is None or result.keypoints is None:
            return []
        ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().numpy()
        xy = result.keypoints.xy.cpu().numpy()
        if result.keypoints.conf is not None:
            conf = result.keypoints.conf.cpu().numpy()
        else:
            conf = np.ones(xy.shape[:2])
        return [
            Person(track_id=track_id, xy=xy[i], conf=conf[i], box=tuple(float(v) for v in boxes[i]))
            for i, track_id in enumerate(ids)
        ]

    def warm_up(self) -> None:
        """Run once on a blank picture: the first call sets up the GPU and takes several seconds."""
        self.infer(np.zeros((384, 640, 3), np.uint8))
        self.reset()

    def reset(self) -> None:
        """Forget all tracks, so a new video starts with fresh worker numbers."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()
