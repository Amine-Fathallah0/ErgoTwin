"""Video input: a webcam, a video file or a network stream, read on its own thread."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import cv2

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

# Ask FFmpeg not to buffer network streams, so the picture stays close to live.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay")


def parse_source(spec: str, root: Path) -> tuple[str, int | str]:
    """Turns what the user typed into ("webcam", number), ("stream", url) or ("file", path)."""
    spec = spec.strip()
    if spec.isdigit():
        return "webcam", int(spec)
    if "://" in spec:
        return "stream", spec
    path = Path(spec)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"No such video file: {path}")
    return "file", str(path)


class FrameSource:
    """Keeps only the newest frame, so a slow consumer never falls behind a live feed.

    A video file is played at its own speed and restarted at the end, so it
    behaves like a camera pointed at the same scene.
    """

    def __init__(self, spec: str, root: Path):
        self.spec = spec.strip()
        self.kind, self.target = parse_source(self.spec, root)
        self.status = "connecting"  # then "live", "ended" or "error"
        self.error = ""
        self.loops = 0  # how many times a video file has restarted
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def latest(self):
        """(frame number, newest frame). The number tells the caller whether the frame is new."""
        with self._lock:
            return self._seq, self._frame

    def _open(self) -> cv2.VideoCapture:
        if self.kind == "webcam":
            cap = cv2.VideoCapture(self.target, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        if self.kind == "stream":
            return cv2.VideoCapture(self.target, cv2.CAP_FFMPEG)
        return cv2.VideoCapture(self.target)

    def _run(self) -> None:
        try:
            cap = self._open()
        except Exception as exc:  # a bad URL can raise instead of returning a closed capture
            self.status, self.error = "error", str(exc)
            return
        if not cap.isOpened():
            self.status, self.error = "error", f"Could not open {self.spec}"
            cap.release()
            return

        period = 0.0  # live sources set their own pace
        if self.kind == "file":
            fps = cap.get(cv2.CAP_PROP_FPS)
            period = 1.0 / fps if 1.0 <= fps <= 120.0 else 1.0 / 25.0
        next_t = time.perf_counter()
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if self.kind != "file":
                        self.status, self.error = "ended", "The stream stopped."
                        return
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # play the file again from the start
                    ok, frame = cap.read()
                    if not ok:
                        self.status, self.error = "error", f"Could not read {self.spec}"
                        return
                    self.loops += 1
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                self.status = "live"
                if period:
                    next_t += period
                    delay = next_t - time.perf_counter()
                    if delay > 0:
                        self._stop.wait(delay)
                    elif delay < -1.0:  # fell far behind: do not rush to catch up
                        next_t = time.perf_counter()
        finally:
            cap.release()
