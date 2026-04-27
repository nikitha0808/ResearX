"""Visual paper reviewer.

Three modes (REVIEWER_MODE):

  - computer_use : Anthropic's computer_20251124 tool drives a Playwright-rendered
                   PDF gallery in Chromium. Claude scrolls / zooms / inspects
                   figures and tables, then submits a NeurIPS-style review.
  - pipeline     : Take N upfront viewport screenshots, send them all in one
                   Claude call, then a small CU sub-loop crops the architecture
                   figure tightly. Cheapest visual mode.
  - text         : PyMuPDF text extraction, single Claude call. No vision, no
                   browser. Useful as a baseline and for cost-conscious runs.

Top-level entry point: review_paper() dispatches by REVIEWER_MODE.
"""

from __future__ import annotations

import os
import time

from anthropic import Anthropic

from cu_agent import (
    AGENT_MAX_STEPS,
    COMPUTER_USE_BETA,
    CU_DISPLAY_HEIGHT,
    CU_DISPLAY_WIDTH,
    budget_hint,
    execute_cu_action,
    cu_tool_spec,
)
from helpers import (
    CLAUDE_MODEL,
    Review,
    UsageRecord,
    arxiv_pdf_url,
    as_image_block,
    cost_from_usage,
    extract_pdf_text,
    load_paper_in_browser,
    make_client,
    render_page,
    render_page_crop,
    shared_or_fresh_page,
)
from playwright.sync_api import Page

REVIEWER_MODE = os.environ.get("REVIEWER_MODE", "pipeline")
# pipeline     : N upfront screenshots in one Claude call (default);
#                submit_review returns the figure bbox.
# computer_use : full CU agent reads the PDF visually end-to-end; also
#                returns the figure bbox in submit_review.
# text         : PyMuPDF text extraction, single Claude call, no vision.
#                Hero figure is captured by the separate `find_figures`
#                pipeline step, not here.
MAX_SCREENSHOTS = int(os.environ.get("MAX_SCREENSHOTS", "8"))
FIGURE_FINDER_MAX_STEPS = int(os.environ.get("FIGURE_FINDER_MAX_STEPS", "6"))

# Per-callsite model. Falls back to CLAUDE_MODEL (helpers.py) so existing
# setups that only set CLAUDE_MODEL still work. Override via REVIEWER_MODEL
# in .env to e.g. run reviewer on Opus while synthesis stays on Sonnet.
REVIEWER_MODEL = os.environ.get("REVIEWER_MODEL", CLAUDE_MODEL)


REVIEWER_SYSTEM = """You are a reviewer for a top-tier ML conference (NeurIPS / ICLR / ICML).
You are reading a paper via screenshots of its rendered PDF pages.

Evaluate the paper rigorously using the standard reviewer form. Be a harsh,
honest reviewer — most papers should rate 4-7 out of 10. Only rate 7+ if the
contribution is genuinely strong. Actively look for:

  - unsupported claims (claims not backed by the shown evidence)
  - missing baselines in tables
  - cherry-picked results
  - unclear methodology diagrams
  - overclaiming relative to what the figures actually show

In your weaknesses and rationale, reference specific things you observe in the
paper's figures, tables, and architecture diagrams — not just the prose. This
is the whole point of reading the paper visually rather than just the abstract.

Scoring rubric:
  - soundness (1-4): are the claims supported by evidence shown?
      1=unsupported, 2=weak, 3=solid, 4=strong
  - presentation (1-4): clarity of writing, figures, tables.
  - contribution (1-4): novelty and significance to the field.
  - rating (1-10): overall. 1-3=reject, 4-5=borderline, 6-7=weak accept, 8-10=strong accept.
  - confidence (1-5): How confident you are in your judgment.

BUDGET: Be efficient. After you've seen the abstract, the key methodology figure,
and the main results table — that is usually enough to form a confident verdict.
Most papers should require ~5-7 turns of exploration, not the full budget. Do
not keep scrolling or zooming past what you need; the structured review matters
more than complete coverage. As soon as you have a confident verdict, call
submit_review immediately.

FIGURE PREVIEW: While reading, identify the main architecture / method figure
(usually Figure 1; sometimes Figure 2 if Figure 1 is a teaser plot). In
submit_review, pass:

  - `figure_page_index`: 0-indexed PDF page where the figure lives.
  - `figure_bbox_normalized`: [x1, y1, x2, y2] each in [0,1] giving the
    figure's bounding box ON THAT PAGE (0,0 = top-left of the page). Include
    the caption directly below the figure inside the box.

The system renders the page at high DPI and crops to your bbox to produce a
clean thumbnail — you don't need to use the `zoom` action for this. Be
generous with the bbox: err toward including a bit too much margin rather
than clipping. The most common failure mode is setting y2 too low and cutting
off the bottom of a tall architecture diagram.

If the paper has no clear architecture/method figure (pure-theory paper,
survey, etc.), omit figure_bbox_normalized and just pass figure_page_index=0.

You MUST call the submit_review tool to return your judgment. Do not answer in prose."""


REVIEWER_SYSTEM_TEXT = """You are a reviewer for a top-tier ML conference (NeurIPS / ICLR / ICML).
You are reading a paper as TEXT EXTRACTED from its PDF — figures, tables, and
architecture diagrams are NOT visible to you, only their prose descriptions
(captions and references in the body text).

Evaluate the paper rigorously using the standard reviewer form. Be a harsh,
honest reviewer — most papers should rate 4-6 out of 10. Only rate 7+ if the
contribution is genuinely strong. Look for:

  - unsupported claims (claims the prose makes but doesn't justify)
  - methodology gaps (steps that are vague or under-specified)
  - missing baselines or ablations the text references
  - cherry-picked framing or overclaiming
  - clarity of writing, structure, and argument

Because you cannot see the figures and tables themselves, do NOT speculate about
what they show — judge based on the prose, captions, and described setup. Set
your CONFIDENCE accordingly: 2-3 is normal for a text-only read, since you're
working with partial information. Do not penalize a paper for figures you can't
see; do penalize unclear text references to those figures.

Scoring rubric:
  - soundness (1-4): are the claims supported by evidence as described?
  - presentation (1-4): clarity of writing and argument structure.
  - contribution (1-4): novelty and significance to the field.
  - rating (1-10): overall. 1-3=reject, 4-5=borderline, 6-7=weak accept, 8-10=strong accept.
  - confidence (1-5): How confident you are in your judgment.

You will not see figures — just pass `figure_page_index=0` in submit_review;
the system renders page 1 as a fallback thumbnail.

You MUST call the submit_review tool to return your judgment. Do not answer in prose."""


SUBMIT_REVIEW_TOOL = {
    "name": "submit_review",
    "description": "Submit your structured NeurIPS-style review of the paper.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "1-2 sentences on what the paper does."},
            "strengths": {"type": "array", "items": {"type": "string"}, "description": "2-4 concrete strengths."},
            "weaknesses": {"type": "array", "items": {"type": "string"}, "description": "2-4 concrete weaknesses, citing specific figures/tables where possible."},
            "soundness": {"type": "integer", "minimum": 1, "maximum": 4},
            "presentation": {"type": "integer", "minimum": 1, "maximum": 4},
            "contribution": {"type": "integer", "minimum": 1, "maximum": 4},
            "rating": {"type": "integer", "minimum": 1, "maximum": 10},
            "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
            "rationale": {"type": "string", "description": "One paragraph summarizing the judgment."},
            "figure_page_index": {
                "type": "integer",
                "minimum": 0,
                "description": "0-indexed PDF page where the main architecture/method figure appears.",
            },
            "figure_bbox_normalized": {
                "type": "array",
                "description": (
                    "Bounding box of the main architecture figure ON THE PAGE referenced by figure_page_index, "
                    "given as [x1, y1, x2, y2] each in [0,1] (0,0 = top-left of the page; 1,1 = bottom-right). "
                    "Include the figure's caption directly underneath in the box. "
                    "Be generous — err toward including a bit too much rather than clipping. "
                    "If the paper has no clear architecture figure, omit this field."
                ),
                "items": {"type": "number", "minimum": 0, "maximum": 1},
                "minItems": 4,
                "maxItems": 4,
            },
        },
        "required": ["summary", "strengths", "weaknesses", "soundness", "presentation", "contribution", "rating", "confidence", "rationale"],
    },
}


# ── Figure-finder tool (used by text_with_figure mode) ─────────────────────

REPORT_FIGURE_TOOL = {
    "name": "report_figure",
    "description": "Report the location of the paper's main architecture/method figure (usually Figure 1).",
    "input_schema": {
        "type": "object",
        "properties": {
            "figure_page_index": {
                "type": "integer",
                "minimum": 0,
                "description": "0-indexed PDF page where the figure lives.",
            },
            "figure_bbox_normalized": {
                "type": "array",
                "description": (
                    "Bounding box [x1, y1, x2, y2] each in [0,1] on the page identified by "
                    "figure_page_index (0,0 = top-left). Include the caption directly below "
                    "the figure. Be generous — err toward more margin rather than clipping."
                ),
                "items": {"type": "number", "minimum": 0, "maximum": 1},
                "minItems": 4,
                "maxItems": 4,
            },
        },
        "required": ["figure_page_index"],
    },
}


FIND_FIGURE_SYSTEM = """You are locating ONE figure in a research paper — the main architecture or method diagram (usually Figure 1; sometimes Figure 2 if Figure 1 is just a teaser plot).

The paper is rendered as a vertical scrollable gallery in your browser. Scroll through the first few pages, identify the architecture figure, and call report_figure with its page index and bounding box.

  - figure_page_index: 0-indexed PDF page where the figure lives.
  - figure_bbox_normalized: [x1, y1, x2, y2] each in [0,1] on that page.
    Include the caption directly below. Err toward more margin than clipping.

If the paper has no clear architecture figure (pure-theory paper, survey),
call report_figure with figure_page_index=0 and omit figure_bbox_normalized.

BUDGET: ~6 turns. Call report_figure as soon as you've found it. Do NOT
write a review — figure-finding is your only job here.

You MUST call report_figure to exit. Do not answer in prose."""


def finalize_figure(
    pdf_url: str,
    fig_page: int,
    bbox_norm: list[float] | tuple[float, float, float, float] | None = None,
    zoom_png: bytes | None = None,
) -> bytes | None:
    """Pick the figure_png. Priority order:

      1. If `bbox_norm` is provided and valid, render+crop the page at high DPI
         using PyMuPDF — this is the most accurate path because we control
         rendering and the model only had to identify *where on the page*.
      2. Else if `zoom_png` is provided and is plausibly a real crop (≥ 400×400),
         use it. Kept for backward compat with CU `zoom` action captures.
      3. Else render the full page as a fallback thumbnail.
    """
    import struct as _struct

    if bbox_norm is not None:
        cropped = render_page_crop(pdf_url, fig_page, tuple(bbox_norm))
        if cropped is not None:
            try:
                cw, ch = _struct.unpack(">II", cropped[16:24])
            except Exception:
                cw = ch = 0
            print(f"    figure: bbox crop {cw}x{ch} ({len(cropped)} bytes, page {fig_page}, bbox {tuple(bbox_norm)})")
            return cropped
        print(f"    figure: bbox crop failed (page {fig_page}, bbox {tuple(bbox_norm)}) — trying fallbacks")

    MIN_W, MIN_H = 400, 400
    if zoom_png is not None:
        try:
            zw, zh = _struct.unpack(">II", zoom_png[16:24])
        except Exception:
            zw = zh = 0
        if zw >= MIN_W and zh >= MIN_H:
            print(f"    figure: zoom capture {zw}x{zh} ({len(zoom_png)} bytes, page {fig_page})")
            return zoom_png
        print(f"    figure: zoom capture {zw}x{zh} too small (<{MIN_W}x{MIN_H})")

    rendered = render_page(pdf_url, page_index=fig_page)
    print(f"    figure: page-{fig_page} full render ({len(rendered) if rendered else 0} bytes)")
    return rendered


# ── Mode 1: pipeline ────────────────────────────────────────────────────────

def capture_screenshots_pipeline(page: Page, max_shots: int) -> list[bytes]:
    shots: list[bytes] = []
    for _ in range(max_shots):
        shots.append(page.screenshot(full_page=False))
        page.keyboard.press("PageDown")
        page.wait_for_timeout(400)
    return shots


def review_pipeline(client: Anthropic, topic: str, title: str, abstract: str, pdf_url: str) -> tuple[Review, UsageRecord, bytes | None]:
    t0 = time.time()
    with shared_or_fresh_page(viewport={"width": 1280, "height": 1600}) as page:
        n_pages = load_paper_in_browser(page, pdf_url)
        # Don't shoot more screenshots than there are pages — for a short
        # paper the trailing shots are just empty canvas at the bottom of
        # the gallery, wasting tokens and time.
        n_shots = max(1, min(MAX_SCREENSHOTS, n_pages))
        shots = capture_screenshots_pipeline(page, n_shots)

        content: list[dict] = [{
            "type": "text",
            "text": (
                f"TOPIC: {topic}\n\n"
                f"TITLE: {title}\n\n"
                f"ABSTRACT:\n{abstract}\n\n"
                f"The {len(shots)} images below are consecutive pages/viewports of the paper's PDF, in order. "
                f"Read the figures and tables, not just the text."
            ),
        }]
        content.extend(as_image_block(s) for s in shots)

        resp = client.messages.create(
            model=REVIEWER_MODEL,
            max_tokens=2000,
            system=[{"type": "text", "text": REVIEWER_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            tools=[SUBMIT_REVIEW_TOOL],
            tool_choice={"type": "tool", "name": "submit_review"},
            messages=[{"role": "user", "content": content}],
        )

        review = next((b.input for b in resp.content if b.type == "tool_use" and b.name == "submit_review"), None)
        if review is None:
            raise RuntimeError(f"Model did not call submit_review. stop_reason={resp.stop_reason}")

    # Crop the architecture figure deterministically using the bbox the model
    # reported in submit_review. No second Claude call needed.
    fig_page = int(review.get("figure_page_index", 0) or 0)
    figure_png = finalize_figure(pdf_url, fig_page, bbox_norm=review.get("figure_bbox_normalized"))

    usage: UsageRecord = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cache_read_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        "cost_usd": cost_from_usage(REVIEWER_MODEL, resp.usage),
        "seconds": time.time() - t0,
        "screenshots": len(shots),
    }
    return review, usage, figure_png


# ── Mode 2: computer_use (Anthropic computer_20251124) ─────────────────────

def review_computer_use(client, topic: str, title: str, abstract: str, pdf_url: str) -> tuple[Review, UsageRecord, bytes | None]:
    """Agent-loop variant using Anthropic's official computer_20251124 tool.

    Same API surface as the official Docker reference implementation; differs
    only in the execution substrate (Playwright instead of xdotool/Xvfb).
    """
    t0 = time.time()
    totals = {"in": 0, "out": 0, "cr": 0, "cw": 0, "cost": 0.0, "shots": 0}
    step = 0
    last_zoom_png: bytes | None = None

    with shared_or_fresh_page(viewport={"width": CU_DISPLAY_WIDTH, "height": CU_DISPLAY_HEIGHT}) as page:
        load_paper_in_browser(page, pdf_url)

        first_png = page.screenshot(full_page=False)
        totals["shots"] += 1

        cu_tool = cu_tool_spec()

        messages: list[dict] = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"TOPIC: {topic}\n\nTITLE: {title}\n\nABSTRACT:\n{abstract}\n\n"
                        f"A browser is open showing the paper's PDF rendered as a vertical scrollable gallery. "
                        f"Use the computer tool to read it: scroll down to see more pages, press Page_Down for "
                        f"page-by-page navigation, or call zoom on a specific figure/table region (with [x1,y1,x2,y2]) "
                        f"to read it at full resolution. Examine figures, tables, and ablation studies — that's the "
                        f"whole point of reviewing visually. When you reach the main architecture / method figure "
                        f"(usually Figure 1), call zoom once with a tight bounding box around it — that capture "
                        f"becomes the paper's preview thumbnail. Then continue and call submit_review with your "
                        f"verdict and `figure_page_index` set to the page where that figure lives."
                    ),
                },
                as_image_block(first_png),
            ],
        }]

        review: Review | None = None
        for step in range(AGENT_MAX_STEPS):
            # In the final 2 turns, drop the CU tool so the model is forced to
            # submit_review. Cleanest hard stop without fighting tool_choice.
            final_turns = step >= AGENT_MAX_STEPS - 2
            tools_for_step = [SUBMIT_REVIEW_TOOL] if final_turns else [cu_tool, SUBMIT_REVIEW_TOOL]

            resp = client.beta.messages.create(
                model=REVIEWER_MODEL,
                max_tokens=2000,
                system=[{"type": "text", "text": REVIEWER_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                tools=tools_for_step,
                messages=messages,
                betas=[COMPUTER_USE_BETA],
            )
            totals["in"] += resp.usage.input_tokens
            totals["out"] += resp.usage.output_tokens
            totals["cr"] += getattr(resp.usage, "cache_read_input_tokens", 0) or 0
            totals["cw"] += getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
            totals["cost"] += cost_from_usage(REVIEWER_MODEL, resp.usage)

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                break

            messages.append({"role": "assistant", "content": resp.content})

            tool_results: list[dict] = []
            for block in tool_uses:
                if block.name == "submit_review":
                    review = block.input
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Review received. Session ending.",
                    })
                elif block.name == "computer":
                    png, is_error = execute_cu_action(page, block.input)
                    totals["shots"] += 1
                    if block.input.get("action") == "zoom" and not is_error:
                        last_zoom_png = png
                    content_blocks: list[dict] = []
                    hint = budget_hint(step, AGENT_MAX_STEPS)
                    if hint:
                        content_blocks.append({"type": "text", "text": hint})
                    content_blocks.append(as_image_block(png))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content_blocks,
                        **({"is_error": True} if is_error else {}),
                    })

            messages.append({"role": "user", "content": tool_results})

            if review is not None:
                break

    if review is None:
        raise RuntimeError(f"computer-use agent ended without submit_review (after {step + 1} steps)")

    usage: UsageRecord = {
        "input_tokens": totals["in"],
        "output_tokens": totals["out"],
        "cache_read_tokens": totals["cr"],
        "cache_write_tokens": totals["cw"],
        "cost_usd": totals["cost"],
        "seconds": time.time() - t0,
        "screenshots": totals["shots"],
    }
    fig_page = int(review.get("figure_page_index", 0) or 0)
    figure_png = finalize_figure(
        pdf_url,
        fig_page,
        bbox_norm=review.get("figure_bbox_normalized"),
        zoom_png=last_zoom_png,
    )
    return review, usage, figure_png


# ── Mode 3: text-only (no vision, no browser) ──────────────────────────────

def review_text_only(client, topic: str, title: str, abstract: str, pdf_url: str) -> tuple[Review, UsageRecord, bytes | None]:
    """Text-only review: extract paper text via PyMuPDF, no images, single Claude call.

    Useful as the cheapest mode and as a baseline for the "what does visual
    reasoning actually add?" comparison.
    """
    t0 = time.time()
    full_text, n_pages = extract_pdf_text(pdf_url)

    user_msg = (
        f"TOPIC: {topic}\n\n"
        f"TITLE: {title}\n\n"
        f"ABSTRACT:\n{abstract}\n\n"
        f"FULL PAPER TEXT (extracted from PDF, {n_pages} pages — figures/tables not visible):\n\n"
        f"{full_text}"
    )

    resp = client.messages.create(
        model=REVIEWER_MODEL,
        max_tokens=2000,
        system=[{"type": "text", "text": REVIEWER_SYSTEM_TEXT, "cache_control": {"type": "ephemeral"}}],
        tools=[SUBMIT_REVIEW_TOOL],
        tool_choice={"type": "tool", "name": "submit_review"},
        messages=[{"role": "user", "content": user_msg}],
    )

    review = next((b.input for b in resp.content if b.type == "tool_use" and b.name == "submit_review"), None)
    if review is None:
        raise RuntimeError(f"text-only reviewer didn't call submit_review. stop_reason={resp.stop_reason}")

    usage: UsageRecord = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cache_read_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        "cost_usd": cost_from_usage(REVIEWER_MODEL, resp.usage),
        "seconds": time.time() - t0,
        "screenshots": 0,
    }
    # Text-only mode produces no figure here. The separate find_figures
    # pipeline step in main.py runs the CU figure-finder on kept papers.
    return review, usage, None


# ── Public: CU sub-loop for hero-figure capture (used by main.py) ──────────

def find_hero_figure_cu(client, pdf_url: str) -> tuple[bytes | None, UsageRecord]:
    """CU sub-loop that locates a paper's main architecture figure and returns
    a high-DPI crop of it. Used by text_with_figure mode so the cheap text
    review can still ship with a real figure thumbnail.

    Bounded to FIGURE_FINDER_MAX_STEPS turns. The agent is given a single goal
    tool (report_figure) — it scrolls the PDF gallery, identifies the figure,
    and returns its page index + normalized bbox. The crop is then produced
    deterministically by render_page_crop, same as in pipeline / CU modes.
    """
    t0 = time.time()
    totals = {"in": 0, "out": 0, "cr": 0, "cw": 0, "cost": 0.0, "shots": 0}
    figure_info: dict | None = None

    with shared_or_fresh_page(viewport={"width": CU_DISPLAY_WIDTH, "height": CU_DISPLAY_HEIGHT}) as page:
        load_paper_in_browser(page, pdf_url)
        first_png = page.screenshot(full_page=False)
        totals["shots"] += 1

        cu_tool = cu_tool_spec()

        messages: list[dict] = [{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Find the paper's main architecture / method figure. The paper is "
                    "rendered as a scrollable gallery below. Scroll through the first few "
                    "pages, identify the architecture figure (usually Figure 1), then call "
                    "report_figure with its page index and bounding box."
                )},
                as_image_block(first_png),
            ],
        }]

        for step in range(FIGURE_FINDER_MAX_STEPS):
            # In the final 2 turns, drop the CU tool so the model is forced to
            # call report_figure. Same hard-stop pattern as review_computer_use.
            final_turns = step >= FIGURE_FINDER_MAX_STEPS - 2
            tools_for_step = [REPORT_FIGURE_TOOL] if final_turns else [cu_tool, REPORT_FIGURE_TOOL]

            resp = client.beta.messages.create(
                model=REVIEWER_MODEL,
                max_tokens=1000,
                system=[{"type": "text", "text": FIND_FIGURE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                tools=tools_for_step,
                messages=messages,
                betas=[COMPUTER_USE_BETA],
            )
            totals["in"] += resp.usage.input_tokens
            totals["out"] += resp.usage.output_tokens
            totals["cr"] += getattr(resp.usage, "cache_read_input_tokens", 0) or 0
            totals["cw"] += getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
            totals["cost"] += cost_from_usage(REVIEWER_MODEL, resp.usage)

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                break

            messages.append({"role": "assistant", "content": resp.content})

            tool_results: list[dict] = []
            for block in tool_uses:
                if block.name == "report_figure":
                    figure_info = block.input
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Figure recorded. Session ending.",
                    })
                elif block.name == "computer":
                    png, is_error = execute_cu_action(page, block.input)
                    totals["shots"] += 1
                    content_blocks: list[dict] = []
                    if is_error:
                        content_blocks.append({"type": "text", "text": "(action errored; current state below)"})
                    content_blocks.append(as_image_block(png))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content_blocks,
                    })

            messages.append({"role": "user", "content": tool_results})

            if figure_info is not None:
                break

    usage: UsageRecord = {
        "input_tokens": totals["in"],
        "output_tokens": totals["out"],
        "cache_read_tokens": totals["cr"],
        "cache_write_tokens": totals["cw"],
        "cost_usd": totals["cost"],
        "seconds": time.time() - t0,
        "screenshots": totals["shots"],
    }

    if figure_info is None:
        # Agent never reported a figure — fall back to page-1 render so the
        # paper still gets a thumbnail, just not a tight crop.
        return render_page(pdf_url, page_index=0), usage

    fig_page = int(figure_info.get("figure_page_index", 0) or 0)
    figure_png = finalize_figure(pdf_url, fig_page, bbox_norm=figure_info.get("figure_bbox_normalized"))
    return figure_png, usage


# ── Dispatcher ─────────────────────────────────────────────────────────────

def review_paper(topic: str, title: str, arxiv_url: str, abstract: str) -> tuple[Review, UsageRecord, bytes | None]:
    """Produce a structured review of one paper. Strategy is chosen by REVIEWER_MODE.

    Returns (review, usage, figure_png_or_None). Text mode always returns
    figure_png=None — the separate find_figures pipeline step handles hero
    figure capture. Pipeline / computer_use modes still bundle the figure
    bbox into submit_review so they return a real crop here for free.
    """
    client = make_client()
    pdf_url = arxiv_pdf_url(arxiv_url)
    if REVIEWER_MODE == "computer_use":
        fn = review_computer_use
    elif REVIEWER_MODE == "text":
        fn = review_text_only
    else:
        fn = review_pipeline
    return fn(client, topic, title, abstract, pdf_url)


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python reviewer.py <arxiv_id_or_url> [topic] [figure_out.png]")
        sys.exit(1)

    arxiv_ref = sys.argv[1]
    topic = sys.argv[2] if len(sys.argv) > 2 else "general machine learning"
    figure_out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/review_figure.png")

    review, usage, figure_png = review_paper(
        topic=topic,
        title="(fetched from arxiv via smoke-test)",
        arxiv_url=arxiv_ref,
        abstract="(abstract not provided in smoke test — reviewer should read the paper itself)",
    )
    if figure_png:
        figure_out.write_bytes(figure_png)
    print(json.dumps({
        "review": review,
        "usage": usage,
        "figure_png_bytes": len(figure_png) if figure_png else 0,
        "figure_path": str(figure_out) if figure_png else None,
    }, indent=2, default=str))
