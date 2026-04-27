"""Cross-paper synthesis: given N reviewed papers, propose new research ideas.

The cognitive heart of the ideation pipeline. Takes structured reviews from
reviewer.py and asks Claude to find non-obvious synergies — methods from one
paper applied to another's domain, gaps in one paper that another solves, etc.
Output is a structured list of Idea dicts that downstream posters consume.

Filtering: ideas referencing fewer than 2 source papers are dropped. The whole
point is cross-paper synthesis; single-paper extensions belong elsewhere.
"""

from __future__ import annotations

import os
import time
from typing import TypedDict

from helpers import CLAUDE_MODEL, UsageRecord, cost_from_usage, make_client

# Per-callsite model. Falls back to CLAUDE_MODEL (helpers.py) when unset.
# Override via SYNTHESIZE_MODEL in .env to e.g. run synthesis on Opus while
# the reviewer stays on Sonnet — synthesis is one call per topic, so the
# cost delta is small and the quality lift is worth it.
SYNTHESIZE_MODEL = os.environ.get("SYNTHESIZE_MODEL", CLAUDE_MODEL)

SYNTHESIZE_SYSTEM = """You are a senior ML researcher brainstorming a research
agenda. You are given reviews of N recent papers on a single topic. Propose
1-3 NEW research ideas that combine insights from multiple papers in
non-obvious ways.

Each idea MUST:
  - Cite at least TWO source papers by their 0-indexed position.
  - Address a CONCRETE problem (not just "could be useful").
  - Have an experimental plan that's actually doable in 3-6 months.
  - Be NOVEL — don't restate what any source paper already does.

Look for SYNERGIES, not stacking. Examples of good ideas:

  - "Paper A's training method applied to paper B's task, where the failure
     mode B describes is exactly what A is designed to fix."
  - "Paper A solves the data problem that paper B explicitly cites as its
     biggest limitation in the discussion."
  - "Combining A's eval framework with B's models reveals a regression
     hidden in B's reported results."

Bad ideas to avoid:

  - "Do X but with transformers / RL / a bigger model" — that's stacking, not synergy.
  - "Combine A and B" without articulating the SPECIFIC connection that
     makes the combination interesting.
  - Ideas where the source papers don't actually constrain or enable the proposal.

Be honest about feasibility. If you can only find 1 strong cross-paper idea,
propose 1. Quality over quantity.

For each idea, also assign a NOVELTY RATING (integer 1-10) using this rubric:
  1-3   Low novelty — incremental, mostly stacking; the combination is obvious
        or already widely studied.
  4-6   Moderate novelty — interesting combination, but somewhat predictable
        in hindsight; either source paper might already gesture at it.
  7-8   Strong novelty — real cross-paper synergy that neither source addresses
        alone; the connection isn't obvious until pointed out.
  9-10  Surprising / breakthrough-level synergy — the combination unlocks
        something neither field would have arrived at by itself.

Be calibrated, not generous. Most synergies fall in the 4-7 range.

Call propose_ideas with your output. Do not answer in prose."""

PROPOSE_IDEAS_TOOL = {
    "name": "propose_ideas",
    "description": "Submit your proposed research ideas synthesizing the input papers.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ideas": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Short, specific title (~10 words)."},
                        "source_paper_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "description": "0-indexed positions of source papers (at least 2).",
                        },
                        "problem": {"type": "string", "description": "1-2 sentences naming the concrete problem this addresses."},
                        "novelty": {"type": "string", "description": "1-2 sentences on the specific synergy — what does combining the sources enable that none alone does?"},
                        "novelty_rating": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "1-10 score for how novel the synergy is. See system prompt for rubric.",
                        },
                        "experimental_plan": {"type": "string", "description": "2-4 sentences of concrete first experiments to validate the idea."},
                        "impact": {"type": "string", "description": "1 sentence: if this works, what does it unlock?"},
                    },
                    "required": ["title", "source_paper_indices", "problem", "novelty", "novelty_rating", "experimental_plan", "impact"],
                },
            },
        },
        "required": ["ideas"],
    },
}


class Idea(TypedDict, total=False):
    title: str
    source_paper_indices: list[int]
    problem: str
    novelty: str
    novelty_rating: int
    experimental_plan: str
    impact: str
    # Filled in by synthesize_ideas after the call:
    source_paper_urls: list[str]
    source_paper_titles: list[str]


_EMPTY_USAGE: UsageRecord = {
    "input_tokens": 0, "output_tokens": 0,
    "cache_read_tokens": 0, "cache_write_tokens": 0,
    "cost_usd": 0.0, "seconds": 0.0, "screenshots": 0,
}


def synthesize_ideas(papers_with_reviews: list[dict], topic: str) -> tuple[list[Idea], UsageRecord]:
    """Given papers with reviews attached, propose cross-paper research ideas.

    Each entry in papers_with_reviews must have keys: title, url, summary, review.
    The review must have: summary, strengths, weaknesses, rationale.
    """
    if len(papers_with_reviews) < 2:
        return [], _EMPTY_USAGE

    t0 = time.time()
    client = make_client()

    paper_blocks: list[str] = []
    for i, p in enumerate(papers_with_reviews):
        r = p["review"]
        paper_blocks.append(
            f"[Paper {i}] {p['title']}\n"
            f"  arxiv: {p['url']}\n"
            f"  Summary: {r['summary']}\n"
            f"  Strengths:\n" + "\n".join(f"    - {s}" for s in r["strengths"]) + "\n"
            f"  Weaknesses:\n" + "\n".join(f"    - {w}" for w in r["weaknesses"]) + "\n"
            f"  Reviewer's take: {r['rationale']}"
        )
    user_msg = f"Topic: {topic}\n\nReviewed papers:\n\n" + "\n\n".join(paper_blocks)

    resp = client.messages.create(
        model=SYNTHESIZE_MODEL,
        max_tokens=3000,
        system=[{"type": "text", "text": SYNTHESIZE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        tools=[PROPOSE_IDEAS_TOOL],
        # `any` (rather than the more restrictive `tool` choice) requires the
        # model to call a tool but lets it emit reasoning text first. With
        # `tool`, Claude often skips straight to the tool call with no prose.
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": user_msg}],
    )

    # Print any free-text reasoning the model produced before the tool call,
    # so the synthesis step is observable from the terminal.
    for content_block in resp.content:
        if content_block.type == "text" and content_block.text.strip():
            print("  [synthesize] reasoning:")
            for line in content_block.text.strip().splitlines():
                print(f"    {line}")

    block = next((b for b in resp.content if b.type == "tool_use" and b.name == "propose_ideas"), None)
    if block is None:
        raise RuntimeError(f"synthesizer didn't call propose_ideas. stop_reason={resp.stop_reason}")

    raw_ideas = block.input["ideas"]
    ideas: list[Idea] = []
    for raw in raw_ideas:
        indices = sorted({i for i in raw["source_paper_indices"] if 0 <= i < len(papers_with_reviews)})
        if len(indices) < 2:
            continue  # synthesis ideas must reference ≥2 sources
        ideas.append({
            **raw,
            "source_paper_indices": indices,
            "source_paper_urls": [papers_with_reviews[i]["url"] for i in indices],
            "source_paper_titles": [papers_with_reviews[i]["title"] for i in indices],
        })

    usage: UsageRecord = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cache_read_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        "cost_usd": cost_from_usage(SYNTHESIZE_MODEL, resp.usage),
        "seconds": time.time() - t0,
        "screenshots": 0,
    }
    return ideas, usage
