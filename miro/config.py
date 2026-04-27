"""Miro configuration: env vars, layout constants, file paths.

All hardcoded constants and environment-driven knobs live here. Other miro/
modules import from this one so values are defined once.
"""

from __future__ import annotations

import os
from pathlib import Path

from helpers import CLAUDE_MODEL

# ── REST endpoint + auth ────────────────────────────────────────────────────

MIRO_API = "https://api.miro.com/v2"
MIRO_TOKEN = os.environ.get("MIRO_TOKEN", "").strip()

# ── Backend selection ──────────────────────────────────────────────────────

MIRO_BACKEND = os.environ.get("MIRO_BACKEND", "cu").lower()  # cu | rest
MIRO_FALLBACK_TO_REST = os.environ.get("MIRO_FALLBACK_TO_REST", "1") == "1"

# ── File paths (state + auth session) ──────────────────────────────────────
# Project root is two levels up from this file: miro/config.py → miro/ → repo.
# Pipeline state files live under .state/ (gitignored). The Miro Playwright
# session stays at the repo root next to .env — it's a credential file, not
# pipeline state, so it shares its sibling's home.

_REPO_ROOT = Path(__file__).resolve().parent.parent
MIRO_STATE_FILE = _REPO_ROOT / ".state" / "miro_state.json"
MIRO_SESSION_FILE = _REPO_ROOT / ".miro_session.json"

# ── Layout ──────────────────────────────────────────────────────────────────
# Miro uses center-origin per item, board canvas is unbounded. Papers render
# as colored circles scattered around the canvas (matches the look in
# miro_sample.png). Ideas remain yellow sticky notes.

PAPER_CIRCLE_DIAM = 240
PAPER_SCATTER_RADIUS = 700  # bounding radius for spring layout (board units)
PAPER_FILL_COLORS = [
    "#2D9BF0",  # blue
    "#5DCC52",  # green
    "#C09EFF",  # purple
    "#00B7B0",  # teal
    "#C5E373",  # lime
    "#FFC54A",  # amber  (extra slots for >5 papers)
    "#FF6B6B",  # red
]
IDEA_STICKY_W = 240

# Connector style — the reference image has visible, weighty edges.
CONNECTOR_STROKE_WIDTH = "4"
CONNECTOR_STROKE_COLOR = "#1a1a1a"

# ── CU agent ───────────────────────────────────────────────────────────────
# Per-item sub-loop: each sticky gets its own bounded CU session. This caps
# token usage per call (no quadratic history bloat) and keeps the per-minute
# rate limit happy.

MIRO_CU_VIEWPORT_W = int(os.environ.get("MIRO_CU_VIEWPORT_W", "1280"))
MIRO_CU_VIEWPORT_H = int(os.environ.get("MIRO_CU_VIEWPORT_H", "1600"))
MIRO_CU_PER_ITEM_STEPS = int(os.environ.get("MIRO_CU_PER_ITEM_STEPS", "10"))
MIRO_CU_PACE_SECONDS = int(os.environ.get("MIRO_CU_PACE_SECONDS", "4"))

# Miro CU agents are the most demanding visual task in the pipeline (web-app
# DOM with overlays, fragile drag/click sequences), so historically this
# defaulted to Opus while the rest of the pipeline ran on Sonnet. We now
# fall back to CLAUDE_MODEL so a single env var sets the whole pipeline's
# model; override MIRO_CU_MODEL in .env to bump just this stage to Opus.
MIRO_CU_MODEL = os.environ.get("MIRO_CU_MODEL", CLAUDE_MODEL)
