"""Miro REST API layer + deterministic backend.

This module:
  - owns the REST verbs (board/items/connectors GET/POST)
  - provides idempotent placement (cached state IDs validated against live board)
  - hosts the NetworkX-driven layout (used by both backends to keep idea
    stickies near their source-paper circles)
  - implements `post_topic_via_rest`, the deterministic backend and the silent
    fallback when CU crashes
"""

from __future__ import annotations

import re

import networkx as nx
import requests

from helpers import warn_exception

from .config import (
    CONNECTOR_STROKE_COLOR,
    CONNECTOR_STROKE_WIDTH,
    IDEA_STICKY_W,
    MIRO_API,
    MIRO_TOKEN,
    PAPER_CIRCLE_DIAM,
    PAPER_FILL_COLORS,
    PAPER_SCATTER_RADIUS,
)
from .state import idea_hash, items_for_board, load_state, save_state


# ── REST plumbing ──────────────────────────────────────────────────────────

def rest_headers() -> dict:
    return {
        "Authorization": f"Bearer {MIRO_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def check_token() -> None:
    if not MIRO_TOKEN:
        raise RuntimeError("MIRO_TOKEN must be set in .env")


def rest_post(path: str, payload: dict) -> dict:
    """POST to a Miro REST endpoint. `path` is everything after the API base."""
    r = requests.post(f"{MIRO_API}/{path}", json=payload, headers=rest_headers(), timeout=20)
    if not r.ok:
        # Surface the API error message — the default raise_for_status hides it.
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise requests.HTTPError(f"{r.status_code} {r.reason} on {path}: {detail}", response=r)
    return r.json()


# ── Board creation ─────────────────────────────────────────────────────────

def ensure_board_for_topic(topic: str) -> tuple[str, str]:
    """Return (board_id, view_url) for the topic, creating a new board if needed.

    Uses POST /v2/boards. Idempotent — second call with same topic returns
    the cached board_id from .miro_state.json.
    """
    check_token()
    state = load_state()
    boards = state.setdefault("boards", {})

    if topic in boards:
        b = boards[topic]
        return b["id"], b.get("view_url", f"https://miro.com/app/board/{b['id']}/")

    payload = {
        "name": topic[:60],
        "description": f"Auto-created by curator agent for topic: {topic}",
    }
    board = rest_post("boards", payload)
    bid = board["id"]
    view_url = board.get("viewLink", f"https://miro.com/app/board/{bid}/")

    state = load_state()
    state.setdefault("boards", {})[topic] = {"id": bid, "view_url": view_url}
    save_state(state)
    return bid, view_url


# ── Layout ─────────────────────────────────────────────────────────────────

def compute_layout(
    papers: list[dict],
    ideas: list[dict],
    scale: float = PAPER_SCATTER_RADIUS,
) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
    """Compute board positions for papers and ideas via NetworkX spring layout.

    Builds a bipartite graph (papers ↔ ideas via source_paper_indices), runs
    Fruchterman-Reingold to get an organic, edge-crossing-minimizing placement,
    and returns two dicts keyed by index:

        paper_positions[i] = (x, y)   # for papers[i]
        idea_positions[j]  = (x, y)   # for ideas[j]

    Coordinates are scaled to roughly [-scale, +scale] in Miro board space.
    Connected nodes (an idea and its source papers) end up near each other,
    so connectors visually converge — what the reference image shows.

    Falls back to a regular polygon (`circular_layout`) when the graph has no
    edges (e.g. ideas list is empty or has no valid source_paper_indices).
    """
    G = nx.Graph()
    for i in range(len(papers)):
        G.add_node(("paper", i))
    for j, idea in enumerate(ideas):
        G.add_node(("idea", j))
        for src in idea.get("source_paper_indices") or []:
            if 0 <= src < len(papers):
                G.add_edge(("paper", src), ("idea", j))

    if G.number_of_nodes() == 0:
        return {}, {}

    if G.number_of_edges() == 0:
        pos = nx.circular_layout(G, scale=scale)
    else:
        # k controls ideal node spacing; bumped to give idea stickies room
        # between paper circles (centroid-pulled ideas were overlapping P0
        # at lower k values). seed makes layout reproducible per run.
        pos = nx.spring_layout(G, k=2.5, iterations=300, seed=42, scale=scale)

    paper_positions = {
        i: (float(pos[("paper", i)][0]), float(pos[("paper", i)][1]))
        for i in range(len(papers))
    }
    idea_positions = {
        j: (float(pos[("idea", j)][0]), float(pos[("idea", j)][1]))
        for j in range(len(ideas))
    }
    return paper_positions, idea_positions


# ── State / item validation ────────────────────────────────────────────────

def miro_item_exists(board_id: str, item_id: str) -> bool:
    """GET /boards/{id}/items/{item_id} — return True iff the item is still there.

    Used to validate cached state IDs before short-circuiting placement: if a
    prior cleanup pass deleted the item but state still references its ID,
    we treat the state hit as a cache miss and create a fresh one.

    Sentinels and empty IDs are treated as non-existent.
    """
    if not item_id or item_id == "cu_placed":
        return False
    try:
        r = requests.get(
            f"{MIRO_API}/boards/{board_id}/items/{item_id}",
            headers=rest_headers(),
            timeout=10,
        )
        return r.ok
    except Exception as e:
        warn_exception(f"miro_item_exists({item_id})", e)
        return False


def drop_state_entry(board_id: str, kind: str, key: str) -> None:
    """Remove a stale (kind, key) entry from state. kind ∈ {'papers', 'ideas'}."""
    state = load_state()
    items = state.get("items", {}).get(board_id, {})
    if items.get(kind, {}).pop(key, None) is not None:
        save_state(state)


# ── Placement (papers, ideas, connectors) ──────────────────────────────────

def post_paper_circle_rest(
    board_id: str,
    paper: dict,
    position: tuple[float, float],
    color_index: int = 0,
) -> str:
    """Place a paper as a colored Miro circle (shape='circle') with the title inside.

    Replaces the old card-based representation so the board matches the look
    in miro_sample.png (papers as colored circles connected by lines to ideas).
    Idempotent — repeats with the same paper URL return the existing item ID,
    but only if that ID still exists on the board (validated via REST GET).
    """
    state = load_state()
    items = items_for_board(state, board_id)
    url = paper["url"]
    cached = items["papers"].get(url)
    if cached and miro_item_exists(board_id, cached):
        return cached
    if cached:
        # Stale state — item was deleted out from under us. Drop the entry so
        # we don't keep returning a phantom ID, then create fresh below.
        print(f"    [state] paper '{paper['title'][:50]}' cached as {cached} but missing on board — recreating")
        drop_state_entry(board_id, "papers", url)

    color = PAPER_FILL_COLORS[color_index % len(PAPER_FILL_COLORS)]
    title = paper["title"][:80]

    payload = {
        "data": {
            "shape": "circle",
            "content": f"<p><strong>{title}</strong></p>",
        },
        "style": {
            "fillColor": color,
            "borderColor": "#1a1a1a",
            "borderWidth": "2",
            "color": "#1a1a1a",
            "fontSize": "14",
            "textAlign": "center",
            "textAlignVertical": "middle",
        },
        "position": {"x": position[0], "y": position[1], "origin": "center"},
        "geometry": {"width": PAPER_CIRCLE_DIAM, "height": PAPER_CIRCLE_DIAM},
    }
    item_id = rest_post(f"boards/{board_id}/shapes", payload)["id"]

    state = load_state()
    items_for_board(state, board_id)["papers"][url] = item_id
    save_state(state)
    return item_id


def post_idea_sticky_rest(board_id: str, idea: dict, position: tuple[float, float]) -> str:
    state = load_state()
    items = items_for_board(state, board_id)
    h = idea_hash(idea)
    cached = items["ideas"].get(h)
    if cached and miro_item_exists(board_id, cached):
        return cached
    if cached:
        print(f"    [state] idea '{idea['title'][:50]}' cached as {cached} but missing on board — recreating")
        drop_state_entry(board_id, "ideas", h)

    rating = idea.get("novelty_rating")
    rating_line = f"(novelty {rating}/10)\n\n" if rating else ""
    text = (
        f"💡 {idea['title']}\n"
        f"{rating_line}"
        f"PROBLEM: {idea['problem']}\n\n"
        f"NOVELTY: {idea['novelty']}\n\n"
        f"PLAN: {idea['experimental_plan']}\n\n"
        f"IMPACT: {idea.get('impact', '')}"
    )
    # Minimal payload: Miro's sticky_notes endpoint is picky about combinations
    # of data.shape + style.fillColor + geometry.height. Keep it lean — only
    # specify width (height auto-derives), let the default (yellow) shape stand.
    payload = {
        "data": {"content": text[:1500]},
        "style": {"fillColor": "light_yellow"},
        "position": {"x": position[0], "y": position[1], "origin": "center"},
        "geometry": {"width": IDEA_STICKY_W},
    }
    item_id = rest_post(f"boards/{board_id}/sticky_notes", payload)["id"]

    state = load_state()
    items_for_board(state, board_id)["ideas"][h] = item_id
    save_state(state)
    return item_id


def connect_rest(board_id: str, start_id: str, end_id: str, label: str = "") -> str | None:
    """Best-effort connector. Returns None on API error.

    Uses CONNECTOR_STROKE_WIDTH / CONNECTOR_STROKE_COLOR so edges read clearly
    against the colored circles — the reference image (miro_sample.png) shows
    weighty, visible connectors, not hairlines.
    """
    payload = {
        "startItem": {"id": start_id},
        "endItem": {"id": end_id},
        "style": {
            "strokeWidth": CONNECTOR_STROKE_WIDTH,
            "strokeColor": CONNECTOR_STROKE_COLOR,
            "strokeStyle": "normal",
            "startStrokeCap": "none",
            "endStrokeCap": "none",
        },
    }
    if label:
        payload["captions"] = [{"content": label}]
    try:
        return rest_post(f"boards/{board_id}/connectors", payload)["id"]
    except Exception as e:
        warn_exception(f"connect_rest {start_id}->{end_id}", e)
        return None


# ── Board introspection (used by CU phase 4 to find/connect new stickies) ──

def fetch_board_stickies(board_id: str) -> list[dict]:
    """Return all sticky_note items currently on the board (Miro REST GET).

    Used after the CU agent has placed idea stickies — we need their item IDs
    to draw connectors via REST, but CU has no DOM access to return them.
    """
    try:
        r = requests.get(
            f"{MIRO_API}/boards/{board_id}/items",
            params={"type": "sticky_note", "limit": 50},
            headers=rest_headers(),
            timeout=20,
        )
        if not r.ok:
            print(f"  ! [fetch_board_stickies] HTTP {r.status_code}: {r.text[:200]}")
            return []
        return r.json().get("data", []) or []
    except Exception as e:
        warn_exception("fetch_board_stickies", e)
        return []


def fetch_existing_connector_pairs(board_id: str) -> set[tuple[str, str]]:
    """Return the set of (start_id, end_id) pairs for connectors already on
    the board, both directions, so callers can dedupe before creating new ones.

    Miro returns connectors paginated; we walk the cursor until exhausted. If
    any page errors out we return what we have — better than nothing for the
    dedupe check (worst case we create a duplicate connector).
    """
    pairs: set[tuple[str, str]] = set()
    cursor: str | None = None
    while True:
        params: dict = {"limit": 50}
        if cursor:
            params["cursor"] = cursor
        try:
            r = requests.get(
                f"{MIRO_API}/boards/{board_id}/connectors",
                params=params,
                headers=rest_headers(),
                timeout=20,
            )
            if not r.ok:
                print(f"  ! [fetch_existing_connector_pairs] HTTP {r.status_code}: {r.text[:200]}")
                break
            data = r.json()
        except Exception as e:
            warn_exception("fetch_existing_connector_pairs", e)
            break
        for c in data.get("data", []) or []:
            s = (c.get("startItem") or {}).get("id")
            e = (c.get("endItem") or {}).get("id")
            if s and e:
                pairs.add((s, e))
                pairs.add((e, s))
        cursor = data.get("cursor")
        if not cursor:
            break
    return pairs


def match_idea_stickies_rest(board_id: str, ideas: list[dict]) -> dict[int, str]:
    """Match each idea (by index) to the sticky on the board whose content
    contains its title. Returns {idea_index: item_id}.

    Miro returns sticky content as either plain text or HTML — we strip tags
    before substring-matching. Title-match is fuzzy (lowercased prefix) to
    survive minor typing inaccuracies by the CU agent.
    """
    stickies = fetch_board_stickies(board_id)
    if not stickies:
        return {}

    def content_text(s: dict) -> str:
        raw = s.get("data", {}).get("content", "") or ""
        return re.sub(r"<[^>]+>", " ", raw).lower()

    sticky_texts = [(s["id"], content_text(s)) for s in stickies]

    matched: dict[int, str] = {}
    used: set[str] = set()
    for i, idea in enumerate(ideas):
        key = idea["title"][:30].lower()
        if not key:
            continue
        for sid, txt in sticky_texts:
            if sid in used:
                continue
            if key in txt:
                matched[i] = sid
                used.add(sid)
                break
    return matched


# ── Top-level REST backend ─────────────────────────────────────────────────

def post_topic_via_rest(topic: str, papers: list[dict], ideas: list[dict]) -> dict:
    """Deterministic path: papers as colored circles scattered around the canvas,
    idea stickies placed at the centroid of each idea's source papers, connectors
    fanning out from each source paper to its idea(s)."""
    check_token()
    board_id, view_url = ensure_board_for_topic(topic)

    # NetworkX-driven layout: positions for both papers and ideas come from a
    # spring layout over the (paper ↔ idea) edge graph, so connected items end
    # up near each other and edge crossings are minimized.
    paper_positions, idea_positions = compute_layout(papers, ideas)

    paper_id_by_url: dict[str, str] = {}
    for i, p in enumerate(papers):
        x, y = paper_positions.get(i, (0.0, 0.0))
        try:
            pid = post_paper_circle_rest(board_id, p, (x, y), color_index=i)
            paper_id_by_url[p["url"]] = pid
        except requests.HTTPError as e:
            print(f"  ! paper circle failed for '{p['title'][:40]}': {e}")

    idea_ids: list[str] = []
    for j, idea in enumerate(ideas):
        x, y = idea_positions.get(j, (0.0, 0.0))
        try:
            iid = post_idea_sticky_rest(board_id, idea, (x, y))
            idea_ids.append(iid)
        except requests.HTTPError as e:
            print(f"  ! idea sticky failed for '{idea['title'][:40]}': {e}")
            continue

        for src_url in idea.get("source_paper_urls", []):
            src_id = paper_id_by_url.get(src_url)
            if src_id:
                connect_rest(board_id, src_id, iid)

    return {
        "backend": "rest",
        "board_id": board_id,
        "view_url": view_url,
        "paper_ids": list(paper_id_by_url.values()),
        "idea_ids": idea_ids,
        "topic": topic,
    }
