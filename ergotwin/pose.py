"""Pose backend: finds and tracks people with a YOLO pose model, on the GPU when there is one.

To use another pose model, write a class with the same members (`device_name`, `infer`,
`infer_tiles`, `reset`) that returns `Person` objects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import PoseConfig
from .skeleton import Person


TRACK_LOW = 0.1  # a detection this weak may continue a track, never start one


def _tracker_options(cfg: PoseConfig) -> dict:
    """ByteTrack settings, built from config.toml."""
    return {
        "tracker_type": "bytetrack",
        "track_high_thresh": cfg.conf,
        "track_low_thresh": TRACK_LOW,
        "new_track_thresh": cfg.conf,
        "track_buffer": cfg.track_memory_frames,
        "match_thresh": 0.8,
        "fuse_score": True,
    }


def _tracker_settings(cfg: PoseConfig) -> str:
    """The same settings as a file, which is how the library's own tracking takes them."""
    lines = ["# Written by ergotwin/pose.py on every start. Change config.toml, not this file."]
    lines += [f"{key}: {value}" for key, value in _tracker_options(cfg).items()]
    return "\n".join(lines) + "\n"


class YoloPose:
    def __init__(self, cfg: PoseConfig, models_dir: Path, tracker_file: str = "tracker.yaml"):
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
        self._tracker_file = models_dir / tracker_file
        self._tracker_file.write_text(_tracker_settings(cfg), encoding="utf-8")
        self._tile_trackers: list = []  # one per grid tile: each tile is a camera of its own

    def _options(self) -> dict:
        # The model's own cut-off is set low so the tracker also sees weak detections: a worker
        # half hidden for a moment keeps their number. New workers still need cfg.conf.
        precision = {"quantize": 16} if self.half else {}  # 16-bit floats on the GPU
        return dict(device=self.device, imgsz=self.cfg.imgsz, conf=TRACK_LOW, max_det=self.cfg.max_people,
                    verbose=False, **precision)

    def infer(self, frame: np.ndarray) -> list[Person]:
        """People in one BGR frame, each with a track id that stays the same across frames."""
        result = self.model.track(frame, persist=True, tracker=str(self._tracker_file), **self._options())[0]
        return self._people(result)

    def infer_tiles(self, frames: list[np.ndarray]) -> list[list[Person]]:
        """People in several pictures at once (the tiles of a grid), each picture tracked on its own.

        The model looks at all the tiles in one batch, which keeps the graphics card busy. Give
        every tile the same size: tiles of different sizes are padded to squares, which is slower.
        """
        if not frames:
            return []
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace

        # Each tile's results come off the graphics card in one copy, not one small copy per value.
        results = [result.cpu() for result in self.model.predict(frames, **self._options())]
        while len(self._tile_trackers) < len(results):
            self._tile_trackers.append(BYTETracker(IterableSimpleNamespace(**_tracker_options(self.cfg))))
        return [self._people(_follow(tracker, result)) for tracker, result in zip(self._tile_trackers, results)]

    @staticmethod
    def _people(result) -> list[Person]:
        if result is None or result.boxes is None or result.boxes.id is None or result.keypoints is None:
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

    def warm_up(self, tiles: int = 0, size: tuple[int, int] = (640, 384)) -> None:
        """Run once on blank pictures of (width, height): the first call sets up the GPU and takes seconds."""
        blank = np.zeros((size[1], size[0], 3), np.uint8)
        if tiles:
            self.infer_tiles([blank] * tiles)
        else:
            self.infer(blank)
        self.reset()

    def reset(self) -> None:
        """Forget all tracks, so a new video starts with fresh worker numbers."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()
        self._tile_trackers = []


def _follow(tracker, result):
    """Give one picture's detections to its own tracker, as the library's tracking step does.

    Returns the detections that belong to a track, carrying their track ids; None when there are none.
    """
    import torch

    tracks = tracker.update(result.boxes.cpu().numpy(), result.orig_img)
    if len(tracks) == 0:
        return None
    kept = result[tracks[:, -1].astype(int)]  # last column: which detection each track came from
    kept.update(boxes=torch.as_tensor(tracks[:, :-1], device=result.boxes.data.device))
    return kept
