#!/usr/bin/env python3
"""Runtime wrapper for local + Hugging Face Spaces deployment."""

from __future__ import annotations

import os

import uvicorn

from server.app import app


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=port, workers=1)
