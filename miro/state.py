"""State persistence for the Miro pipeline.

State (.miro_state.json) schema:
    {
      "boards": {"topic_name": {"id": "board_id", "view_url": "..."}},
      "items":  {"board_id": {"papers": {"arxiv_url": "item_id"},
                              "ideas":  {"hash":     "item_id"}}}
    }

Maps topic → board_id and persists item IDs scoped per board, so re-runs are
idempotent: the second time you run a topic, the agent reuses the existing
board and skips items already placed on it.
"""

from __future__ import annotations

import hashlib
import json

from .config import MIRO_STATE_FILE


def load_state() -> dict:
    if MIRO_STATE_FILE.exists():
        return json.loads(MIRO_STATE_FILE.read_text())
    return {"boards": {}, "items": {}}


def save_state(state: dict) -> None:
    # Atomic write: serialize to <file>.tmp first, then rename. POSIX rename
    # is atomic on the same filesystem, so a Ctrl+C mid-write leaves either
    # the previous good file or a leftover .tmp — never a half-written
    # miro_state.json that would crash the next load_state() with a JSON
    # decode error.
    MIRO_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = MIRO_STATE_FILE.with_name(MIRO_STATE_FILE.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(MIRO_STATE_FILE)


def items_for_board(state: dict, board_id: str) -> dict:
    """Get (or initialize) the per-board items namespace."""
    return state.setdefault("items", {}).setdefault(board_id, {"papers": {}, "ideas": {}})


def idea_hash(idea: dict) -> str:
    key = (idea["title"] + idea.get("problem", ""))[:300]
    return hashlib.md5(key.encode()).hexdigest()[:12]
