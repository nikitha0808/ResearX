"""Miro board posting — top-level dispatcher.

Each topic gets its own dedicated Miro board, created on demand. Two paths
to put items on a board:

  - REST  (deterministic): hits the Miro v2 REST API directly. Reliable.
    Used when MIRO_BACKEND=rest, and as the silent fallback if CU fails.
  - CU    (computer-use agent): drives miro.com via Playwright + Anthropic's
    computer_20251124 tool. The headline demo. Less reliable on canvas
    drag interactions; auto-falls-back to REST on exception.

Public API:
  - post_topic_to_miro — top-level dispatcher (chooses backend by env)
  - MIRO_SESSION_FILE  — Playwright session path (main.py preloads it)
"""

from __future__ import annotations

from .config import MIRO_BACKEND, MIRO_SESSION_FILE
from .cu import post_topic_via_cu
from .rest import post_topic_via_rest

__all__ = ["MIRO_BACKEND", "MIRO_SESSION_FILE", "post_topic_to_miro"]


def post_topic_to_miro(topic: str, papers: list[dict], ideas: list[dict]) -> dict:
    """Dispatch by MIRO_BACKEND. Each topic gets its own dedicated board."""
    if MIRO_BACKEND == "rest":
        return post_topic_via_rest(topic, papers, ideas)
    return post_topic_via_cu(topic, papers, ideas)
