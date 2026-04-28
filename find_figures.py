"""Post-review figure extraction.

Runs only on KEEP_TOP survivors of `review_visual`. A CU agent reads the paper
visually and reports the page index + bounding box of the main architecture /
method figure via the `report_figure` tool. The actual crop is then produced
deterministically by render_page_crop (PyMuPDF, high DPI) — the model only has
to identify *where on the page*, not capture the pixels itself. This keeps the
thumbnail crisp and matches what the model intended.

Falls back to a page-1 render if the agent never finds a figure (e.g. survey
or pure-theory paper).
"""

from __future__ import annotations

import os
import time

from cu_agent import (
    COMPUTER_USE_BETA,
    CU_DISPLAY_HEIGHT,
    CU_DISPLAY_WIDTH,
    budget_hint,
    cu_tool_spec,
    execute_cu_action,
)
from helpers import (
    CLAUDE_MODEL,
    UsageRecord,
    arxiv_pdf_url,
    as_image_block,
    cost_from_usage,
    load_paper_in_browser,
    make_client,
    render_page,
    render_page_crop,
    shared_or_fresh_page,
)

FIGURE_FINDER_MAX_STEPS = int(os.environ.get("FIGURE_FINDER_MAX_STEPS", "8"))
FIGURE_FINDER_MODEL = os.environ.get("FIGURE_FINDER_MODEL", CLAUDE_MODEL)


REPORT_FIGURE_TOOL = {
    "name": "report_figure",
    "description": "Report the location of the paper's main architecture/method figure.",
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
                    "Bounding box [x1, y1, x2, y2] each in [0,1] on the page (0,0 = "
                    "top-left, 1,1 = bottom-right). Include the caption directly below "
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

The paper is rendered as a vertical scrollable gallery in your browser. Use the computer tool to scroll through the first few pages, identify the architecture figure, and call report_figure with its page index and bounding box.

  - figure_page_index: 0-indexed PDF page where the figure lives.
  - figure_bbox_normalized: [x1, y1, x2, y2] each in [0,1] on that page. Include
    the caption directly below the figure. Be generous with the bbox — err toward
    including too much margin rather than clipping. The most common failure mode
    is setting y2 too low and cutting off the bottom of a tall diagram.

If the paper has no clear architecture figure (pure-theory paper, survey, etc.),
call report_figure with figure_page_index=0 and omit figure_bbox_normalized.

BUDGET: ~6 turns. Scroll, locate the figure, report it. Do NOT review the paper
or comment on its content — figure-finding is your only job.

You MUST call report_figure to exit. Do not answer in prose."""


def find_main_figure(arxiv_url: str) -> tuple[bytes | None, UsageRecord]:
    """CU sub-loop that locates a paper's main architecture figure and returns
    a high-DPI crop of it. Bounded to FIGURE_FINDER_MAX_STEPS turns.

    Returns (figure_png, usage). figure_png falls back to a page-1 render if
    the agent doesn't report a usable bbox; only None if PDF download itself
    fails.
    """
    client = make_client()
    pdf_url = arxiv_pdf_url(arxiv_url)
    t0 = time.time()
    totals = {"in": 0, "out": 0, "cr": 0, "cw": 0, "cost": 0.0, "shots": 0}
    figure_info: dict | None = None
    step = 0

    with shared_or_fresh_page(viewport={"width": CU_DISPLAY_WIDTH, "height": CU_DISPLAY_HEIGHT}) as page:
        n_pages = load_paper_in_browser(page, pdf_url)
        if n_pages == 0:
            raise RuntimeError(f"PDF rendered 0 pages: {pdf_url}")

        first_png = page.screenshot(full_page=False)
        totals["shots"] += 1

        cu_tool = cu_tool_spec()

        messages: list[dict] = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Find the paper's main architecture / method figure. The paper is "
                        "rendered as a scrollable gallery below. Scroll through the first few "
                        "pages, identify the architecture figure (usually Figure 1), then call "
                        "report_figure with its page index and a tight bounding box that "
                        "includes the caption."
                    ),
                },
                as_image_block(first_png),
            ],
        }]

        for step in range(FIGURE_FINDER_MAX_STEPS):
            # Final 2 turns: drop the CU tool to force a report_figure call.
            # Same hard-stop pattern as review_computer_use.
            final_turns = step >= FIGURE_FINDER_MAX_STEPS - 2
            tools_for_step = [REPORT_FIGURE_TOOL] if final_turns else [cu_tool, REPORT_FIGURE_TOOL]

            resp = client.beta.messages.create(
                model=FIGURE_FINDER_MODEL,
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
            totals["cost"] += cost_from_usage(FIGURE_FINDER_MODEL, resp.usage)

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
                    hint = budget_hint(step, FIGURE_FINDER_MAX_STEPS)
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

    # Crop deterministically from the bbox the agent reported. Fall back to a
    # page-level render if the bbox is missing or invalid.
    if figure_info is None:
        return render_page(pdf_url, page_index=0), usage

    page_idx = int(figure_info.get("figure_page_index", 0) or 0)
    bbox = figure_info.get("figure_bbox_normalized")
    if bbox is not None:
        cropped = render_page_crop(pdf_url, page_idx, tuple(bbox))
        if cropped is not None:
            return cropped, usage
    return render_page(pdf_url, page_index=page_idx), usage


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python find_figures.py <arxiv_id_or_url> [out.png]")
        sys.exit(1)

    arxiv_ref = sys.argv[1]
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/figure.png")

    figure_png, usage = find_main_figure(arxiv_ref)
    if figure_png:
        out_path.write_bytes(figure_png)
        print(f"figure: {len(figure_png)} bytes → {out_path}")
    else:
        print("figure: unavailable")
    print(f"usage: ${usage['cost_usd']:.3f}, {usage['seconds']:.0f}s, {usage['screenshots']} screenshots")
