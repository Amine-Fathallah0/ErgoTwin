"""Start the camera module:  python -m ergotwin  [--source videos/demo.mp4]"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .config import load_config
from .server import create_app


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(prog="ergotwin", description="ErgoTwin camera module")
    parser.add_argument("--source", help="start right away on a webcam number, a video file or a stream URL")
    parser.add_argument("--config", default=str(root / "config.toml"), help="settings file")
    parser.add_argument("--host", default="127.0.0.1",
                        help="use 0.0.0.0 to let other devices on the network open the page")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(load_config(args.config), root, args.source)
    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"ErgoTwin camera module: open http://{shown}:{args.port}  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", timeout_graceful_shutdown=2)


if __name__ == "__main__":
    main()
