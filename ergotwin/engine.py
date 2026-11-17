"""Runs the pipeline on its own thread: frames in, annotated picture and posture state out."""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path

import cv2

from . import draw
from .config import Config
from .pipeline import Pipeline
from .sources import FrameSource


def _fit(frame, max_width: int):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    return cv2.resize(frame, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)


class Engine:
    def __init__(self, cfg: Config, root: Path):
        self.cfg = cfg
        self.root = root
        self.privacy = False  # draw skeletons on black instead of on the video
        self._pose = None  # loaded on the first start, then reused
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._control = threading.Lock()  # one start or stop at a time
        self._lock = threading.Lock()  # guards the three fields below
        self._jpeg: bytes | None = None
        self._seq = 0
        self._state: dict = {}
        self._set("idle", "No source running. Pick one and press Start.")

    # --- control -------------------------------------------------------------

    def start(self, spec: str) -> None:
        with self._control:
            self._halt()
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(spec, self._stop), daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._control:
            self._halt()
            self._set("idle", "Stopped. Pick a source and press Start.")

    def _halt(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=15)
        self._thread = None

    # --- output --------------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)

    def jpeg(self) -> tuple[int, bytes | None]:
        """(picture number, newest annotated picture as JPEG, or None when nothing is running)."""
        with self._lock:
            return self._seq, self._jpeg

    def _set(self, status: str, message: str, source: str | None = None) -> None:
        with self._lock:
            self._state = {
                "status": status, "message": message, "source": source, "fps": 0.0,
                "device": self._pose.device_name if self._pose else "", "model": self.cfg.pose.model,
                "privacy": self.privacy, "seated": self.cfg.station.seated, "time": 0.0,
                "bad_hold_s": self.cfg.timing.bad_hold_s, "caution_hold_s": self.cfg.timing.caution_hold_s,
                "camera": {"learned": False, "tilt_deg": 0}, "workers": [], "flags": [],
            }
            self._jpeg = None
            self._seq += 1

    # --- the loop ------------------------------------------------------------

    def _run(self, spec: str, stop: threading.Event) -> None:
        source = None
        try:
            self._set("loading", "Loading the pose model...", spec)
            if self._pose is None:
                from .pose import YoloPose

                pose = YoloPose(self.cfg.pose, self.root / "models")
                pose.warm_up()
                self._pose = pose
            self._pose.reset()
            source = FrameSource(spec, self.root)
            source.start()
            self._set("connecting", f"Opening {spec} ...", spec)
            self._loop(spec, stop, source)
        except Exception as exc:
            traceback.print_exc()
            self._set("error", str(exc), spec)
        finally:
            if source is not None:
                source.stop()

    def _loop(self, spec: str, stop: threading.Event, source: FrameSource) -> None:
        cfg, pose = self.cfg, self._pose
        pipeline = Pipeline(cfg)
        started = time.monotonic()
        last_seq, fps, last_done = -1, 0.0, None

        # A video file that restarts is treated as one continuous scene, as a camera would
        # give: workers who stand where they stood keep their number and their timers.
        while not stop.is_set():
            if source.status in ("error", "ended"):
                self._set(source.status, source.error, spec)
                return
            seq, frame = source.latest()
            if frame is None or seq == last_seq:
                stop.wait(0.002)
                continue
            last_seq = seq

            frame = _fit(frame, cfg.pose.max_width)
            t = time.monotonic() - started
            people = pose.infer(frame)
            views = pipeline.process(t, people, (frame.shape[1], frame.shape[0]), clock=time.strftime("%H:%M:%S"))

            picture = draw.render(frame, people, views, cfg.pose.keypoint_conf, self.privacy)
            ok, encoded = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 80])
            now = time.monotonic()
            if last_done is not None and now > last_done:
                rate = 1.0 / (now - last_done)
                fps = rate if fps == 0.0 else 0.9 * fps + 0.1 * rate
            last_done = now
            if not ok:
                continue
            with self._lock:
                self._jpeg = encoded.tobytes()
                self._seq += 1
                self._state = {
                    "status": "running", "message": "", "source": spec, "fps": round(fps, 1),
                    "device": pose.device_name, "model": cfg.pose.model, "privacy": self.privacy,
                    "seated": cfg.station.seated, "time": round(t, 1),
                    "bad_hold_s": cfg.timing.bad_hold_s, "caution_hold_s": cfg.timing.caution_hold_s,
                    # How far the camera looks down, worked out from the people in view.
                    "camera": {"learned": pipeline.camera.learned, "tilt_deg": round(pipeline.camera.pitch_deg)},
                    "workers": [views[k] for k in sorted(views)],
                    "flags": list(pipeline.flags)[-50:][::-1],  # newest first
                }
