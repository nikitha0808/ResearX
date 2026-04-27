"""Shared utilities used across the pipeline.

What lives here, by category:

  - Backend selection: make_client picks Bedrock or direct Anthropic API.
  - Pricing: cost_from_usage normalizes a usage object into a dollar cost.
  - Image plumbing: as_image_block builds the Anthropic content block.
  - PDF helpers: arxiv_pdf_url / extract_pdf_text / fetch_and_render_pages /
    load_paper_in_browser. PyMuPDF + Playwright primitives shared by the
    reviewer (visual modes), the figure detector, and the text reviewer.
  - Type aliases: Review and UsageRecord, used by reviewer and downstream nodes.
  - Misc: append_jsonl, environment-driven config constants.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import TypedDict

import pymupdf
import requests
from contextlib import contextmanager

from anthropic import Anthropic, AnthropicBedrock
from playwright.sync_api import BrowserContext, Page, sync_playwright

# ── Config ──────────────────────────────────────────────────────────────────

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
PREFILTER_MODEL = os.environ.get("PREFILTER_MODEL", "claude-haiku-4-5-20251001")
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "12"))
RENDER_DPI = int(os.environ.get("RENDER_DPI", "120"))
HEADLESS = os.environ.get("HEADLESS", "0") == "1"

# Pricing (USD per 1M tokens). Update if Anthropic changes.
PRICE_PER_M = {
    "claude-opus-4-7":           {"in": 15.0, "out": 75.0, "cache_read": 1.5,  "cache_write": 18.75},
    "claude-sonnet-4-6":         {"in": 3.0,  "out": 15.0, "cache_read": 0.3,  "cache_write": 3.75},
    "claude-haiku-4-5-20251001": {"in": 1.0,  "out": 5.0,  "cache_read": 0.1,  "cache_write": 1.25},
}


# ── Type aliases ────────────────────────────────────────────────────────────

class Review(TypedDict):
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    soundness: int
    presentation: int
    contribution: int
    rating: int
    confidence: int
    rationale: str


class UsageRecord(TypedDict):
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    seconds: float
    screenshots: int


# ── Logging ─────────────────────────────────────────────────────────────────

def warn_exception(label: str, exc: BaseException) -> None:
    """Print a one-line, consistent exception warning that doesn't disrupt
    the pipeline. Use in defensive try/except blocks where the failure is
    recoverable but the cause should still be visible during demos."""
    print(f"  ! [{label}] {type(exc).__name__}: {exc}")


# ── Backend selection ──────────────────────────────────────────────────────

def make_client():
    """Pick Bedrock if AWS creds are present, else direct Anthropic API."""
    if os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"):
        # AnthropicBedrock reads AWS_REGION / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
        # / AWS_SESSION_TOKEN from env automatically via boto3.
        return AnthropicBedrock()
    return Anthropic()


# ── Shared Playwright context ──────────────────────────────────────────────
#
# main.py opens ONE BrowserContext per topic (with the Miro storage_state
# applied so the CU phase has its session) and registers it here. Reviewer
# and miro request a page via shared_or_fresh_page(), which opens a new tab
# inside that context. Result: a single browser window with one tab per task,
# instead of N separate windows.
#
# Standalone callers (e.g. `python reviewer.py <arxiv-id>`) hit the fallback
# path and get an isolated browser+context — no behavior change there.

_SHARED_CONTEXT: BrowserContext | None = None


def set_shared_context(ctx: BrowserContext | None) -> None:
    """Register (or clear) a BrowserContext whose pages all live as tabs in
    the same browser window. main.py calls this at topic start."""
    global _SHARED_CONTEXT
    _SHARED_CONTEXT = ctx


@contextmanager
def shared_or_fresh_page(
    viewport: dict | None = None,
    storage_state: str | None = None,
):
    """Yield a Playwright Page. If a shared context is registered, the page
    is a new tab in it (and `storage_state` is ignored — the parent context
    already owns its session). Otherwise a fresh browser+context+page is
    launched and torn down on exit."""
    if _SHARED_CONTEXT is not None:
        page = _SHARED_CONTEXT.new_page()
        if viewport:
            try:
                page.set_viewport_size(viewport)
            except Exception:
                pass
        try:
            yield page
        finally:
            try:
                page.close()
            except Exception:
                pass
        return

    # Standalone: build a one-shot stack. This path means main.py's shared
    # context was never registered (e.g. running `python reviewer.py <id>`).
    print("    [browser] no shared context registered — launching standalone browser")
    with sync_playwright() as p:
        b = p.chromium.launch(headless=HEADLESS)
        try:
            ctx = b.new_context(
                viewport=viewport or {"width": 1280, "height": 800},
                storage_state=storage_state,
            )
            try:
                yield ctx.new_page()
            finally:
                ctx.close()
        finally:
            b.close()


# ── Pricing ─────────────────────────────────────────────────────────────────

def cost_from_usage(model: str, usage) -> float:
    # Match by substring so Bedrock IDs like "us.anthropic.claude-opus-4-7-..."
    # resolve to the same pricing row as the direct API name.
    p = next((v for k, v in PRICE_PER_M.items() if k in model or model in k), None)
    if not p:
        if "opus" in model: p = PRICE_PER_M["claude-opus-4-7"]
        elif "sonnet" in model: p = PRICE_PER_M["claude-sonnet-4-6"]
        elif "haiku" in model: p = PRICE_PER_M["claude-haiku-4-5-20251001"]
        else: return 0.0
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    inp = usage.input_tokens - cw - cr
    return (inp * p["in"] + usage.output_tokens * p["out"] + cr * p["cache_read"] + cw * p["cache_write"]) / 1_000_000


# ── Anthropic content blocks ────────────────────────────────────────────────

def as_image_block(png_bytes: bytes) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(png_bytes).decode(),
        },
    }


# ── PDF / arXiv helpers ─────────────────────────────────────────────────────

PREVIEW_DPI = int(os.environ.get("PREVIEW_DPI", "130"))


def arxiv_pdf_url(arxiv_id_or_url: str) -> str:
    m = re.search(r"(\d{4}\.\d{4,5})", arxiv_id_or_url)
    return f"https://arxiv.org/pdf/{m.group(1)}" if m else arxiv_id_or_url


def open_pdf(pdf_url: str) -> pymupdf.Document | None:
    try:
        pdf_bytes = requests.get(pdf_url, timeout=30).content
        return pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        warn_exception(f"open_pdf {pdf_url}", e)
        return None


def render_page(pdf_url: str, page_index: int = 0, dpi: int = PREVIEW_DPI) -> bytes | None:
    """Render a specific page of the PDF as PNG bytes. Page 0 = first page."""
    doc = open_pdf(pdf_url)
    if doc is None:
        return None
    try:
        if page_index < 0 or page_index >= len(doc):
            page_index = 0
        return doc[page_index].get_pixmap(dpi=dpi).tobytes("png")
    finally:
        doc.close()


def render_page_crop(
    pdf_url: str,
    page_index: int,
    bbox_norm: tuple[float, float, float, float],
    dpi: int = 180,
) -> bytes | None:
    """Render a sub-region of a PDF page at high DPI.

    `bbox_norm` is `(x1, y1, x2, y2)` each in [0,1], where (0,0) is the page's
    top-left and (1,1) the bottom-right. The model identifies the figure's
    location at this normalized scale; we do the pixel math + crop deterministically
    so the thumbnail matches what the model intended.

    Returns None on invalid bbox, missing page, or PDF download failure.
    """
    if bbox_norm is None or len(bbox_norm) != 4:
        return None
    x1, y1, x2, y2 = (max(0.0, min(1.0, float(v))) for v in bbox_norm)
    if x2 <= x1 or y2 <= y1:
        return None

    doc = open_pdf(pdf_url)
    if doc is None:
        return None
    try:
        if page_index < 0 or page_index >= len(doc):
            return None
        page = doc[page_index]
        rect = page.rect
        clip = pymupdf.Rect(
            rect.x0 + x1 * rect.width,
            rect.y0 + y1 * rect.height,
            rect.x0 + x2 * rect.width,
            rect.y0 + y2 * rect.height,
        )
        return page.get_pixmap(dpi=dpi, clip=clip).tobytes("png")
    finally:
        doc.close()


def extract_pdf_text(pdf_url: str) -> tuple[str, int]:
    """Pull text out of every page of the PDF (no rendering). Returns (joined_text, page_count)."""
    pdf_bytes = requests.get(pdf_url, timeout=30).content
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    n = min(MAX_PDF_PAGES, len(doc))
    parts = [f"--- Page {i + 1} ---\n{doc[i].get_text()}" for i in range(n)]
    doc.close()
    return "\n".join(parts), n


def fetch_and_render_pages(pdf_url: str) -> list[bytes]:
    """Download the PDF and render its pages to PNG bytes via PyMuPDF.

    We go through PyMuPDF rather than Chromium's PDF viewer because Playwright's
    bundled Chromium treats arxiv PDFs as downloads (no inline viewer). PyMuPDF
    gives us pixel-identical rendering that we then display in an HTML page so
    the agent still reads via a real browser — see load_paper_in_browser.
    """
    pdf_bytes = requests.get(pdf_url, timeout=30).content
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pngs: list[bytes] = []
    for i in range(min(MAX_PDF_PAGES, len(doc))):
        pix = doc[i].get_pixmap(dpi=RENDER_DPI)
        pngs.append(pix.tobytes("png"))
    doc.close()
    return pngs


def load_paper_in_browser(page: Page, pdf_url: str) -> int:
    """Render PDF pages, embed in a scrollable HTML page, open it in the browser.
    Returns the number of pages rendered."""
    pngs = fetch_and_render_pages(pdf_url)
    imgs_html = "".join(
        f'<img src="data:image/png;base64,{base64.b64encode(p).decode()}" '
        f'style="display:block;margin:0 auto 24px;max-width:100%;box-shadow:0 2px 8px rgba(0,0,0,.4)">'
        for p in pngs
    )
    html = (
        "<!doctype html><html><head>"
        "<style>body{margin:0;background:#2a2a2a;padding:24px;font-family:sans-serif}"
        ".label{color:#888;text-align:center;font-size:12px;margin-bottom:8px}</style>"
        "</head><body>"
        f'<div class="label">Paper rendered for agent review — {len(pngs)} pages</div>'
        f"{imgs_html}"
        "</body></html>"
    )
    page.set_content(html, wait_until="load")
    page.wait_for_timeout(400)
    return len(pngs)


# ── Misc ────────────────────────────────────────────────────────────────────

def append_jsonl(path: Path, obj: dict) -> None:
    """Append a JSON line to a file, dropping any bytes-valued keys (e.g. figure_png)."""
    safe = {k: v for k, v in obj.items() if not isinstance(v, bytes)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(safe, default=str) + "\n")
