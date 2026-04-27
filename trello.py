"""Trello operations.

Owns:
  - Authentication / board + list resolution.
  - Polling state for the "new list = new topic" trigger (.seen_lists.json).
  - Card creation: paper cards with full review-form descriptions, plus
    image attachments for paper preview figures. Ideas live on Miro, not here.

Used by main.py — main owns the LangGraph node functions; this module exposes
atomic operations they call.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

TRELLO_KEY = os.environ.get("TRELLO_KEY", "")
TRELLO_TOKEN = os.environ.get("TRELLO_TOKEN", "")
TRELLO_BOARD_NAME = os.environ.get("TRELLO_BOARD_NAME", "Papers Arena")
TRELLO_API = "https://api.trello.com/1"
AUTH = {"key": TRELLO_KEY, "token": TRELLO_TOKEN}

ROOT = Path(__file__).parent
SEEN_FILE = ROOT / ".state" / "seen_lists.json"


# ── Board / list operations ────────────────────────────────────────────────

def resolve_board_id(name: str, create_if_missing: bool = True) -> str:
    """Look up a Trello board by name. Auto-create on first run when
    `create_if_missing` is True (default), so a new user only needs the
    .env tokens — no manual board setup. The new board starts empty (no
    default 'To Do / Doing / Done' lists) so each list the user adds is
    unambiguously a topic trigger."""
    if not TRELLO_KEY or not TRELLO_TOKEN:
        raise SystemExit(
            "TRELLO_KEY and TRELLO_TOKEN must be set in .env. "
            "See README setup steps (trello.com/app-key)."
        )
    r = requests.get(
        f"{TRELLO_API}/members/me/boards",
        params={**AUTH, "fields": "id,name"},
        timeout=10,
    )
    r.raise_for_status()
    for b in r.json():
        if b["name"] == name:
            return b["id"]

    if not create_if_missing:
        raise SystemExit(f"Trello board '{name}' not found on this account.")

    print(f"Trello board '{name}' not found on this account — creating it…")
    resp = requests.post(
        f"{TRELLO_API}/boards/",
        params={**AUTH, "name": name, "defaultLists": "false"},
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()
    bid = body["id"]
    url = body.get("url") or f"https://trello.com/b/{bid}"
    print(f"  -> created '{name}' (id={bid}). Add a list at {url} to trigger a topic run.")
    return bid


def fetch_lists(board_id: str) -> list[dict]:
    r = requests.get(
        f"{TRELLO_API}/boards/{board_id}/lists",
        params={**AUTH, "fields": "id,name"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# ── Polling state ──────────────────────────────────────────────────────────

def load_seen() -> set[str]:
    if SEEN_FILE.exists():
        return set(json.loads(SEEN_FILE.read_text()))
    return set()


def save_seen(seen: set[str]) -> None:
    # Atomic write — see miro/state.py:save_state for the rationale. Same
    # pattern: tmp file, then rename, so a crash never leaves a half-written
    # seen_lists.json that would crash the next load_seen() on JSON decode.
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SEEN_FILE.with_name(SEEN_FILE.name + ".tmp")
    tmp.write_text(json.dumps(sorted(seen)))
    tmp.replace(SEEN_FILE)


# ── Card creation ──────────────────────────────────────────────────────────

def post_paper_card(list_id: str, paper: dict) -> str:
    """Create a paper card with full review-form description.

    If paper has a `figure_png` (bytes), it's attached as the card cover. Returns
    the new card's Trello ID.
    """
    r = paper["review"]
    desc = (
        f"**Rating:** {r['rating']}/10  |  "
        f"**Soundness:** {r['soundness']}/4  |  "
        f"**Contribution:** {r['contribution']}/4  |  "
        f"**Presentation:** {r['presentation']}/4  |  "
        f"**Confidence:** {r['confidence']}/5\n\n"
        f"_{r['rationale']}_\n\n"
        f"**Strengths:**\n" + "\n".join(f"- {s}" for s in r["strengths"]) + "\n\n"
        f"**Weaknesses:**\n" + "\n".join(f"- {w}" for w in r["weaknesses"]) + "\n\n"
        f"**Authors:** {', '.join(paper.get('authors', []))}\n"
        f"**Published:** {paper.get('published', 'n/a')}\n\n"
        f"---\n\n"
        f"**Abstract:** {paper.get('summary', '')}"
    )
    resp = requests.post(
        f"{TRELLO_API}/cards",
        params={**AUTH, "idList": list_id, "name": paper["title"], "desc": desc, "urlSource": paper["url"]},
        timeout=10,
    )
    resp.raise_for_status()
    card_id = resp.json()["id"]

    figure_png = paper.get("figure_png")
    if figure_png:
        try:
            requests.post(
                f"{TRELLO_API}/cards/{card_id}/attachments",
                params={**AUTH, "name": "figure.png", "setCover": "true"},
                files={"file": ("figure.png", figure_png, "image/png")},
                timeout=15,
            ).raise_for_status()
        except requests.HTTPError as e:
            print(f"    ! couldn't attach figure to '{paper['title'][:40]}': {e}")
    return card_id


def post_topic_to_trello(list_id: str, papers: list[dict]) -> None:
    """Post one card per paper to the named Trello list. Trello is the
    'papers to read later' surface — ideas live on Miro, not here."""
    for p in papers:
        try:
            post_paper_card(list_id, p)
        except requests.HTTPError as e:
            print(f"    ! couldn't post paper card '{p['title'][:40]}': {e}")
    print(f"  -> posted {len(papers)} paper card(s)")
