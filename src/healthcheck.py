#!/usr/bin/env python3
# Copyright (c) 2026 Kishore Sridhar - 611451003 and Farhan Hikmatullah Daulay - 611451002
# Tatung University - I4210 AI實務專題
"""src/healthcheck.py - minimal /healthz endpoint for the inference container."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("HEALTHZ_PORT", "8000"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")

def _current_power_mode() -> str:
    """Best-effort read of the live nvpmodel state."""
    try:
        out = subprocess.run(
            ["nvpmodel", "-q"], capture_output=True, text=True, timeout=2,
        )
        for line in out.stdout.splitlines():
            if "Power Mode" in line:
                return line.split(":", 1)[1].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/healthz":
            self.send_error(404)
            return

        body = json.dumps({
            "status": "healthy",
            "model_version": MODEL_VERSION,
            "power_mode": _current_power_mode(),
        }).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: str) -> None:
        pass

def start_in_thread() -> threading.Thread:
    """Start the healthz server on a daemon thread so it dies with main()."""
    server = HTTPServer(("0.0.0.0", PORT), _Handler) # nosec B104
    t = threading.Thread(
        target=server.serve_forever, daemon=True, name="healthz"
    )
    t.start()
    return t
