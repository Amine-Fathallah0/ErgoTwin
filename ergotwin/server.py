"""Local web server: the page, the live picture, and the posture state as JSON."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .config import Config
from .engine import Engine
from .sources import VIDEO_SUFFIXES

STATIC = Path(__file__).parent / "static"


class StartRequest(BaseModel):
    source: str


class SettingsRequest(BaseModel):
    privacy: bool | None = None
    seated: bool | None = None
    bad_hold_s: float | None = None  # flag a bad posture held this long
    caution_hold_s: float | None = None  # flag a caution posture held this long


def _placeholder(text: str) -> bytes:
    """A dark picture with a line of text, shown while no video is running."""
    image = np.full((540, 960, 3), 24, np.uint8)
    text = text.encode("ascii", "replace").decode()[:70]  # OpenCV's built-in font is ASCII only
    cv2.putText(image, text, (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", image)[1].tobytes()


def create_app(cfg: Config, root: Path, autostart: str | None = None) -> FastAPI:
    engine = Engine(cfg, root)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if autostart:
            engine.start(autostart)
        yield
        engine.stop()

    app = FastAPI(title="ErgoTwin camera module", lifespan=lifespan)
    # Lets a dashboard served from another address (another port, or a file) read the data.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/")
    def page():
        return FileResponse(STATIC / "index.html")

    @app.get("/state")
    def state():
        """Live posture data: one entry per worker, plus the latest flags."""
        return engine.snapshot()

    @app.get("/flags")
    def flags(after: int = 0):
        """Flags numbered above `after`, oldest first, and the newest flag number (0 when there is none).

        A client such as the ankle pod polls this with the last number it handled. A `latest`
        below that number means the app was restarted: start again from 0.
        """
        recent = engine.snapshot()["flags"]  # newest first
        return {
            "latest": recent[0]["seq"] if recent else 0,
            "flags": [f for f in reversed(recent) if f["seq"] > after],
        }

    @app.get("/videos")
    def videos():
        folder = root / "videos"
        if not folder.is_dir():
            return []
        return sorted(f"videos/{p.name}" for p in folder.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)

    @app.post("/start")
    def start(request: StartRequest):
        source = request.source.strip()
        if not source:
            raise HTTPException(400, "Give a source: a webcam number, a video file or a stream URL.")
        engine.start(source)
        return {"ok": True}

    @app.post("/stop")
    def stop():
        engine.stop()
        return {"ok": True}

    @app.post("/settings")
    def settings(request: SettingsRequest):
        if request.privacy is not None:
            engine.privacy = request.privacy
        if request.seated is not None:
            cfg.station.seated = request.seated
        # The timers read these each time they check, so a change applies at once.
        if request.bad_hold_s is not None:
            cfg.timing.bad_hold_s = min(600.0, max(1.0, request.bad_hold_s))
        if request.caution_hold_s is not None:
            cfg.timing.caution_hold_s = min(3600.0, max(1.0, request.caution_hold_s))
        return {"privacy": engine.privacy, "seated": cfg.station.seated,
                "bad_hold_s": cfg.timing.bad_hold_s, "caution_hold_s": cfg.timing.caution_hold_s}

    @app.get("/video")
    async def video():
        """The annotated picture as an MJPEG stream: an <img> tag shows it live."""

        async def pictures():
            last, sent_at, data = -1, 0.0, b""
            while True:
                seq, jpeg = engine.jpeg()
                now = time.monotonic()
                if seq != last:
                    last = seq
                    data = jpeg or _placeholder(engine.snapshot()["message"])
                elif now - sent_at < 0.5:
                    await asyncio.sleep(0.01)
                    continue
                # A browser shows a picture only once the next one starts to arrive,
                # so a picture that is not replaced is sent again twice a second.
                sent_at = now
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")

        return StreamingResponse(pictures(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app
