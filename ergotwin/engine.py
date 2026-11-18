"""Runs the pipeline on its own thread: frames in, annotated picture and posture state out."""

from __future__ import annotations

import statistics
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from . import draw
from .config import Config
from .pipeline import NumberPool, Pipeline
from .skeleton import Person
from .sources import FrameSource
from .tiles import crop_tile, parse_tile, tile_box


def _fit(frame, max_width: int, enlarge: bool = False):
    """Scale a frame wider than max_width down to it. With `enlarge` (a small grid tile), scale up as well."""
    h, w = frame.shape[:2]
    if w == max_width or (w < max_width and not enlarge):
        return frame
    interpolation = cv2.INTER_AREA if w > max_width else cv2.INTER_LINEAR
    return cv2.resize(frame, (max_width, round(h * max_width / w)), interpolation=interpolation)


class SingleScene:
    """The whole picture, or one tile of a grid enlarged: one camera, one pipeline."""

    def __init__(self, cfg: Config, pose, view: str):
        self.cfg, self.pose, self.view = cfg, pose, view
        self.tile = parse_tile(view)
        pose.reset()
        self.pipeline = Pipeline(cfg)

    def step(self, frame: np.ndarray, t: float, clock: str, privacy: bool):
        cfg = self.cfg
        frame = _fit(crop_tile(frame, self.tile), cfg.pose.max_width, enlarge=self.tile is not None)
        people = self.pose.infer(frame)
        views = self.pipeline.process(t, people, (frame.shape[1], frame.shape[0]), clock=clock)
        picture = draw.render(frame, people, views, cfg.pose.keypoint_conf, privacy)
        camera = {"learned": self.pipeline.camera.learned, "tilt_deg": round(self.pipeline.camera.pitch_deg)}
        return picture, [views[k] for k in sorted(views)], list(self.pipeline.flags), camera


class GridScene:
    """Every tile of a grid recording at once.

    Each tile is cut out, enlarged and treated as a camera of its own: its own tracks, its own
    camera angle, its own timers. The model looks at all the tiles in one batch. The people
    found are then drawn back onto the whole grid, and share one set of worker numbers.
    """

    KEY = 1_000_000  # track ids are per tile; tile * KEY + id is unique across the grid

    def __init__(self, cfg: Config, pose, view: str):
        self.cfg, self.pose, self.view = cfg, pose, view
        self.cols, self.rows, _ = parse_tile(view)
        numbers = NumberPool()
        self.pipelines = [Pipeline(cfg, numbers, tile=i) for i in range(1, self.cols * self.rows + 1)]

    def tile_size(self, width: int, height: int) -> tuple[int, int]:
        """One size for every tile, so the model can take them as one batch."""
        tw = self.cfg.grid.tile_width
        return tw, round(tw * (height / self.rows) / (width / self.cols))

    def step(self, frame: np.ndarray, t: float, clock: str, privacy: bool):
        cfg = self.cfg
        h, w = frame.shape[:2]
        tw, th = self.tile_size(w, h)
        boxes = [tile_box(w, h, self.cols, self.rows, i) for i in range(1, len(self.pipelines) + 1)]
        crops = [cv2.resize(frame[y0:y1, x0:x1], (tw, th), interpolation=cv2.INTER_LINEAR)
                 for x0, y0, x1, y1 in boxes]
        found = self.pose.infer_tiles(crops)

        canvas = _fit(frame, cfg.pose.max_width, enlarge=True)
        k = canvas.shape[1] / w  # grid picture -> canvas
        people, views = [], {}
        for i, (pipeline, (x0, y0, x1, y1), tile_people) in enumerate(zip(self.pipelines, boxes, found), start=1):
            tile_views = pipeline.process(t, tile_people, (tw, th), clock=clock)
            scale = np.array([(x1 - x0) / tw * k, (y1 - y0) / th * k])  # enlarged tile -> canvas
            offset = np.array([x0 * k, y0 * k])
            for person in tile_people:
                key = i * self.KEY + person.track_id
                bx1, by1 = np.array(person.box[:2]) * scale + offset
                bx2, by2 = np.array(person.box[2:]) * scale + offset
                people.append(Person(key, person.xy * scale + offset, person.conf, (bx1, by1, bx2, by2)))
                if person.track_id in tile_views:
                    views[key] = tile_views[person.track_id]

        picture = draw.render(canvas, people, views, cfg.pose.keypoint_conf, privacy)
        flags = sorted((f for p in self.pipelines for f in p.flags), key=lambda f: f["seq"])
        learned = [p.camera.pitch_deg for p in self.pipelines if p.camera.learned]
        camera = {"learned": bool(learned), "tilt_deg": round(statistics.median(learned)) if learned else 0}
        return picture, [views[k] for k in sorted(views)], flags, camera


class Engine:
    def __init__(self, cfg: Config, root: Path):
        self.cfg = cfg
        self.root = root
        self.privacy = False  # draw skeletons on black instead of on the video
        self._pose = None  # loaded on the first start, then reused
        self._grid_pose = None  # the model for every tile at once, loaded the first time it is needed
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
                "privacy": self.privacy, "seated": self.cfg.station.seated, "tile": self.cfg.camera.tile, "time": 0.0,
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

    def _scene(self, view: str, spec: str, frame: np.ndarray):
        """A fresh scene for the chosen view: new tracks, new camera angle, new timers."""
        tile = parse_tile(view)
        if tile is None or tile[2] is not None:
            return SingleScene(self.cfg, self._pose, view)
        scene = GridScene(self.cfg, None, view)
        if self._grid_pose is None:
            from .pose import YoloPose

            self._set("loading", "Loading the model for every tile...", spec)
            grid = self.cfg.grid
            pose_cfg = replace(self.cfg.pose, model=grid.model, imgsz=grid.tile_width, conf=grid.conf)
            pose = YoloPose(pose_cfg, self.root / "models", tracker_file="tracker-grid.yaml")
            pose.warm_up(tiles=len(scene.pipelines), size=scene.tile_size(frame.shape[1], frame.shape[0]))
            self._grid_pose = pose
        self._grid_pose.reset()
        scene.pose = self._grid_pose
        return scene

    def _loop(self, spec: str, stop: threading.Event, source: FrameSource) -> None:
        cfg = self.cfg
        scene = None
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
            if scene is None or cfg.camera.tile != scene.view:  # another view: a new scene, with its own camera angle
                scene = self._scene(cfg.camera.tile, spec, frame)

            t = time.monotonic() - started
            picture, workers, flags, camera = scene.step(frame, t, time.strftime("%H:%M:%S"), self.privacy)
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
                    "device": scene.pose.device_name, "model": scene.pose.cfg.model, "privacy": self.privacy,
                    "seated": cfg.station.seated, "tile": scene.view, "time": round(t, 1),
                    "bad_hold_s": cfg.timing.bad_hold_s, "caution_hold_s": cfg.timing.caution_hold_s,
                    # How far the camera looks down, worked out from the people in view.
                    "camera": camera,
                    "workers": workers,
                    "flags": flags[-50:][::-1],  # newest first
                }
