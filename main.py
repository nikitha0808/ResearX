"""Personal research-paper ideation agent — orchestrator.

Flow (LangGraph):

    search_arxiv → review_visual → find_figures → synthesize
                 → post_to_trello → post_to_miro

`find_figures` is a dedicated CU step that runs only on KEEP_TOP survivors —
the agent reads the paper visually, identifies the architecture figure, and
reports a bbox we crop deterministically with PyMuPDF.

This file owns only the orchestration: how the nodes fit together, the
per-node logic that pulls inputs/outputs across the graph, and the Trello-list
polling loop that triggers the pipeline. Domain logic lives in:

  reviewer.py     — paper review (3 modes)
  find_figures.py — post-review CU figure extraction
  synthesize.py   — cross-paper ideation
  trello.py       — Trello board / list / card operations
  miro/           — Miro board posting (CU primary, REST fallback)
  cu_agent.py     — shared computer-use action infrastructure
  helpers.py      — shared utilities (client, pricing, PDF rendering, types)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypedDict

import arxiv
import requests
from dotenv import load_dotenv

# Project modules (trello, miro) snapshot env vars at module load. dotenv must
# run BEFORE those imports or TRELLO_KEY / TRELLO_TOKEN / MIRO_TOKEN are read
# as empty strings and the first REST call fails with a 400.
load_dotenv()

from langgraph.graph import END, START, StateGraph  # noqa: E402

from helpers import (  # noqa: E402
    HEADLESS,
    Review,
    UsageRecord,
    append_jsonl,
    set_shared_context,
)
from playwright.sync_api import sync_playwright  # noqa: E402
from miro import MIRO_BACKEND, MIRO_SESSION_FILE, post_topic_to_miro  # noqa: E402
from find_figures import find_main_figure  # noqa: E402
from reviewer import review_paper  # noqa: E402
from run_log import run_log  # noqa: E402
from setup_miro_login import run_miro_login  # noqa: E402
from synthesize import Idea, synthesize_ideas  # noqa: E402
from trello import (  # noqa: E402
    TRELLO_BOARD_NAME,
    fetch_lists,
    load_seen,
    post_topic_to_trello,
    resolve_board_id,
    save_seen,
)

# ── Config (orchestration-level knobs) ─────────────────────────────────────

SEARCH_MAX = int(os.environ.get("SEARCH_MAX", "8"))
KEEP_TOP = int(os.environ.get("KEEP_TOP", "4"))
MIN_RATING = int(os.environ.get("MIN_RATING", "7"))
MIN_SOUNDNESS = int(os.environ.get("MIN_SOUNDNESS", "3"))
MIN_CONTRIBUTION = int(os.environ.get("MIN_CONTRIBUTION", "3"))

SYNTHESIZE_ENABLED = os.environ.get("SYNTHESIZE_ENABLED", "1") == "1"
FIND_FIGURES_ENABLED = os.environ.get("FIND_FIGURES_ENABLED", "1") == "1"
MIRO_ENABLED = os.environ.get("MIRO_ENABLED", "1") == "1"

ROOT = Path(__file__).parent
REJECTED_LOG = ROOT / "papers" / "rejected.jsonl"
REVIEWS_LOG = ROOT / "papers" / "reviews.jsonl"
IDEAS_LOG = ROOT / "papers" / "ideas.jsonl"
POLL_INTERVAL = 30


# ── State schema ───────────────────────────────────────────────────────────

class Paper(TypedDict, total=False):
    title: str
    authors: list[str]
    published: str
    summary: str
    url: str
    review: Review
    usage: UsageRecord
    figure_png: bytes  # main architecture figure crop (set by find_figures)


class AgentState(TypedDict, total=False):
    topic: str
    list_id: str
    papers: list[Paper]    # all candidates from arXiv
    curated: list[Paper]   # post-threshold survivors, enriched with figure
    ideas: list[Idea]      # synthesized cross-paper ideas


# ── Nodes ──────────────────────────────────────────────────────────────────

def search_arxiv(state: AgentState) -> AgentState:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365)
    date_filter = f"submittedDate:[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"
    query = f"({state['topic']}) AND {date_filter}"
    search = arxiv.Search(
        query=query,
        max_results=SEARCH_MAX,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers: list[Paper] = [
        {
            "title": r.title.strip(),
            "authors": [a.name for a in r.authors],
            "published": r.published.date().isoformat(),
            "summary": r.summary.strip().replace("\n", " "),
            "url": r.entry_id,
        }
        for r in arxiv.Client().results(search)
    ]
    print(f"  fetched {len(papers)} candidates from arXiv")
    return {"papers": papers}


def review_visual(state: AgentState) -> AgentState:
    """Per-candidate review via reviewer.review_paper (mode-dispatched)."""
    kept: list[Paper] = []
    all_usage: list[UsageRecord] = []
    total_cost = 0.0
    t_start = time.time()

    for i, p in enumerate(state["papers"], 1):
        print(f"\n  [{i}/{len(state['papers'])}] reviewing: {p['title'][:70]}")
        try:
            review, usage = review_paper(
                topic=state["topic"],
                title=p["title"],
                arxiv_url=p["url"],
                abstract=p["summary"],
            )
        except Exception as e:
            print(f"    ! review failed ({type(e).__name__}): {e}")
            continue

        all_usage.append(usage)
        total_cost += usage["cost_usd"]

        scored: Paper = {**p, "review": review, "usage": usage}
        accept = (
            review["rating"] >= MIN_RATING
            and review["soundness"] >= MIN_SOUNDNESS
            and review["contribution"] >= MIN_CONTRIBUTION
        )
        verdict = "ACCEPT" if accept else "reject"
        print(
            f"    {verdict} — rating {review['rating']}/10, "
            f"soundness {review['soundness']}/4, contribution {review['contribution']}/4 "
            f"(${usage['cost_usd']:.3f}, {usage['seconds']:.0f}s, {usage['screenshots']} screenshots)"
        )

        append_jsonl(REVIEWS_LOG, {**scored, "topic": state["topic"], "accepted": accept, "at": datetime.now().isoformat()})

        if accept:
            kept.append(scored)

    kept.sort(
        key=lambda x: (
            x["review"]["rating"],
            x["review"]["soundness"] + x["review"]["contribution"],
            x["published"],
        ),
        reverse=True,
    )
    kept = kept[:KEEP_TOP]

    print(
        f"\n  summary: kept {len(kept)} of {len(state['papers'])} — "
        f"threshold rating≥{MIN_RATING}, soundness≥{MIN_SOUNDNESS}, contribution≥{MIN_CONTRIBUTION}"
    )
    print(
        f"  totals: ${total_cost:.2f} across {len(all_usage)} reviews, "
        f"{time.time() - t_start:.0f}s wall, "
        f"{sum(u['input_tokens'] for u in all_usage):,} in / "
        f"{sum(u['output_tokens'] for u in all_usage):,} out tokens"
    )
    return {"curated": kept}


def find_figures(state: AgentState) -> AgentState:
    """CU figure extraction for kept papers — runs only on KEEP_TOP survivors.

    Skipped when FIND_FIGURES_ENABLED=0 or no papers cleared the review
    threshold. Per-paper failures are logged and don't abort the pipeline; the
    paper just ships without a figure thumbnail.
    """
    if not FIND_FIGURES_ENABLED:
        return {}
    curated = state.get("curated", [])
    if not curated:
        return {}

    print(f"\n  finding main figure for {len(curated)} paper(s) via CU…")
    total_cost = 0.0
    t_start = time.time()

    for i, p in enumerate(curated, 1):
        print(f"  [{i}/{len(curated)}] figure: {p['title'][:60]}")
        try:
            figure_png, fig_usage = find_main_figure(p["url"])
        except Exception as e:
            print(f"    ! figure capture failed ({type(e).__name__}): {e}")
            continue
        if figure_png:
            p["figure_png"] = figure_png
        total_cost += fig_usage["cost_usd"]
        print(
            f"    {'captured' if figure_png else 'unavailable'} "
            f"(${fig_usage['cost_usd']:.3f}, {fig_usage['seconds']:.0f}s, "
            f"{fig_usage['screenshots']} screenshots)"
        )
    print(f"  figure capture totals: ${total_cost:.2f}, {time.time() - t_start:.0f}s wall")
    return {"curated": curated}


def synthesize_node(state: AgentState) -> AgentState:
    """Cross-paper ideation. Skipped if SYNTHESIZE_ENABLED=0 or fewer than 2 kept papers."""
    if not SYNTHESIZE_ENABLED:
        return {"ideas": []}
    curated = state.get("curated", [])
    if len(curated) < 2:
        print(f"  synthesis skipped ({len(curated)} kept paper(s); need ≥2)")
        return {"ideas": []}

    print(f"  synthesizing cross-paper ideas from {len(curated)} papers…")
    try:
        ideas, usage = synthesize_ideas(curated, state["topic"])
    except Exception as e:
        print(f"  ! synthesis failed ({type(e).__name__}): {e}")
        return {"ideas": []}

    print(f"  → {len(ideas)} idea(s) proposed (${usage['cost_usd']:.3f}, {usage['seconds']:.0f}s)")
    for j, idea in enumerate(ideas, 1):
        srcs = ", ".join(f"P{i + 1}" for i in idea["source_paper_indices"])
        print(f"    {j}. 💡 {idea['title']}  (sources: {srcs})")
        append_jsonl(IDEAS_LOG, {**idea, "topic": state["topic"], "at": datetime.now().isoformat()})
    return {"ideas": ideas}


def post_to_trello_node(state: AgentState) -> AgentState:
    post_topic_to_trello(state["list_id"], state["curated"])
    return {}


def post_to_miro_node(state: AgentState) -> AgentState:
    if not MIRO_ENABLED:
        return {}
    curated = state.get("curated", [])
    ideas = state.get("ideas", [])
    if not curated and not ideas:
        return {}
    try:
        res = post_topic_to_miro(state["topic"], curated, ideas)
        backend = res.get("backend", "?")
        if backend == "cu":
            print(f"  -> Miro (CU): {res.get('items_placed', 0)}/{res.get('items_attempted', 0)} placed (success={res.get('success')})")
        else:
            print(f"  -> Miro (REST): {len(res.get('paper_ids', []))} papers + {len(res.get('idea_ids', []))} ideas")
        if res.get("view_url"):
            print(f"     board: {res['view_url']}")
    except Exception as e:
        print(f"  ! Miro post failed: {type(e).__name__}: {e}")
    return {}


# ── Graph + Trello-poll loop ───────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("search_arxiv", search_arxiv)
    g.add_node("review_visual", review_visual)
    g.add_node("find_figures", find_figures)
    g.add_node("synthesize", synthesize_node)
    g.add_node("post_to_trello", post_to_trello_node)
    g.add_node("post_to_miro", post_to_miro_node)

    g.add_edge(START, "search_arxiv")
    g.add_edge("search_arxiv", "review_visual")
    g.add_edge("review_visual", "find_figures")
    g.add_edge("find_figures", "synthesize")
    g.add_edge("synthesize", "post_to_trello")
    g.add_edge("post_to_trello", "post_to_miro")
    g.add_edge("post_to_miro", END)
    return g.compile()


def first_run_setup() -> str:
    """Bring a brand-new install to a runnable state.

    - Trello: find the topic-trigger board by name; auto-create it on the
      user's account if it doesn't exist yet.
    - Miro: when the CU backend is selected (default) and no saved Playwright
      session exists, open a browser for a one-time interactive login. This
      is the same flow as `python setup_miro_login.py`, but inlined so a
      first-time user only has to run `python main.py`.

    Returns the resolved Trello board_id.
    """
    board_id = resolve_board_id(TRELLO_BOARD_NAME)  # auto-creates if missing

    if MIRO_ENABLED and MIRO_BACKEND == "cu" and not MIRO_SESSION_FILE.exists():
        print(f"\nNo Miro session at {MIRO_SESSION_FILE}.")
        print("Running one-time Miro login so the CU agent can drive your account.")
        run_miro_login(MIRO_SESSION_FILE)

    return board_id


def main() -> None:
    graph = build_graph()
    board_id = first_run_setup()
    seen = load_seen()

    if not seen:
        seen = {lst["id"] for lst in fetch_lists(board_id)}
        save_seen(seen)
        print(f"Baselined {len(seen)} existing lists on '{TRELLO_BOARD_NAME}'.")

    print(f"Watching '{TRELLO_BOARD_NAME}' every {POLL_INTERVAL}s. Ctrl+C to stop.")
    while True:
        try:
            for lst in fetch_lists(board_id):
                if lst["id"] in seen:
                    continue
                with run_log(lst["name"]) as log_path:
                    print(f"\nNew list detected: '{lst['name']}' — running pipeline…")
                    print(f"  log: {log_path}")
                    # Open ONE browser context for the whole topic. Reviewer and
                    # miro request pages via shared_or_fresh_page(); each task is
                    # a new tab in this context, so all activity stays inside one
                    # browser window. The Miro storage_state is preloaded so the
                    # CU phase has its session; reviewer pages don't care about
                    # those cookies (they navigate to arxiv).
                    storage = str(MIRO_SESSION_FILE) if MIRO_SESSION_FILE.exists() else None
                    with sync_playwright() as pw:
                        browser = pw.chromium.launch(headless=HEADLESS)
                        ctx = browser.new_context(
                            viewport={"width": 1280, "height": 1600},
                            storage_state=storage,
                        )
                        # Keepalive tab: pin the browser window open. Without this,
                        # closing the last task tab also closes the window, and the
                        # next task spawns a brand-new window — so visually it
                        # looks like multiple browsers instead of tabs in one.
                        keepalive = ctx.new_page()
                        keepalive.goto("about:blank")
                        set_shared_context(ctx)
                        try:
                            graph.invoke({"topic": lst["name"], "list_id": lst["id"]})
                        finally:
                            # Best-effort teardown. Ctrl+C mid-pipeline can leave
                            # any of these handles already closed; suppress the
                            # resulting TargetClosedError so the user just sees
                            # a clean exit instead of a Playwright traceback.
                            set_shared_context(None)
                            for closer in (keepalive.close, ctx.close, browser.close):
                                try:
                                    closer()
                                except Exception:
                                    pass
                seen.add(lst["id"])
                save_seen(seen)
        except requests.HTTPError as e:
            print(f"Trello API error: {e}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted — exiting.")
