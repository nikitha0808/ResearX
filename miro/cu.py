"""Computer-use backend for the Miro pipeline.

Drives miro.com via Playwright + Anthropic's computer_20251124 tool. Each
sub-task (place sticky, delete sticky, analyze board) runs as its own
bounded sub-loop with a fresh context, so token usage stays linear in the
number of items and per-minute rate limits don't bite.

The headline flow is `post_topic_via_cu`:

  Phase 1 (CU)   pre-flight cleanup — analyze board, delete stale stickies
  Phase 2 (REST) place colored paper circles via spring layout
  Phase 3 (CU)   place idea stickies between source paper circles
  Phase 4 (REST) match new stickies, draw connectors

Browser-level exceptions silently fall back to `post_topic_via_rest` when
MIRO_FALLBACK_TO_REST=1.
"""

from __future__ import annotations

from pathlib import Path

import requests
from cu_agent import COMPUTER_USE_BETA, execute_cu_action, cu_tool_spec
from helpers import as_image_block, make_client, shared_or_fresh_page

from .config import (
    MIRO_CU_MODEL,
    MIRO_CU_PACE_SECONDS,
    MIRO_CU_PER_ITEM_STEPS,
    MIRO_CU_VIEWPORT_H,
    MIRO_CU_VIEWPORT_W,
    MIRO_FALLBACK_TO_REST,
    MIRO_SESSION_FILE,
)
from .overlays import close_sidekick_panel
from .rest import (
    compute_layout,
    connect_rest,
    fetch_existing_connector_pairs,
    match_idea_stickies_rest,
    post_idea_sticky_rest,
    post_paper_circle_rest,
    ensure_board_for_topic,
    post_topic_via_rest,
)
from .state import idea_hash, items_for_board, load_state, save_state


# ─────────────────────────────────────────────────────────────────────────────
# Sub-loop tools and prompts
# ─────────────────────────────────────────────────────────────────────────────

SUB_DONE_TOOL = {
    "name": "sub_done",
    "description": (
        "Call this exactly once after placing the single sticky note in this turn, "
        "or if you've tried and cannot place it. Always end the sub-session via this tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean", "description": "True if the sticky was placed and contains the requested content."},
            "approx_x": {"type": "integer", "description": "Rough viewport X where the sticky landed (for spatial-awareness in the next sub-session)."},
            "approx_y": {"type": "integer", "description": "Rough viewport Y where the sticky landed."},
            "notes": {"type": "string", "description": "Optional brief note if anything went wrong."},
        },
        "required": ["success"],
    },
}


SUB_LOOP_SYSTEM = """You are placing ONE NEW sticky note on a Miro whiteboard, RIGHT NOW.

You will see TWO images in the user message:
  1. The TARGET LAYOUT — a reference image showing what a completed ideation
     storyboard looks like (papers as colored circles, ideas as yellow stickies,
     thin lines connecting related items). This is your AESTHETIC GOAL.
  2. The CURRENT BOARD STATE — a screenshot of the live Miro board you're
     working on. This is what you actually see.

Place items so the live board moves toward the target style: spread out, no
overlap, with visible structure. Don't try to add lines or change the colors of
your sticky — focus only on placing it cleanly in an empty area.

CRITICAL RULES:
  - You MUST take the placement actions yourself in this session. Existing
    stickies on the board are from PRIOR sessions for OTHER items — they are
    NOT your sticky. Place a NEW sticky for the content the user just gave you.
  - Verify YOUR sticky is on screen by checking that it shows the EXACT
    content from the user message (e.g. "📄 CU smoke paper 3" or
    "💡 CU smoke idea 3"). If the content doesn't match exactly, it's not yours.
  - You MUST call sub_done at the end. If you place the sticky but forget
    sub_done, the system loses your work.

Methods to add a sticky (try in order; stop as soon as ONE works):

  Method A (most reliable): DOUBLE-CLICK an empty area of the canvas.
    Miro pops up a small "create object" widget — click the sticky-note icon
    in that popup. Then type the content directly. Press Escape.

  Method B (toolbar): Look at the LEFT-SIDE TOOLBAR (vertical strip of icons).
    Find the sticky-note icon (yellow square). Click it. A color picker may
    appear — click yellow. Then click on the canvas where you want to drop
    the sticky. Type the content. Press Escape.

  Method C (keyboard): Click ONCE on the canvas to focus it, then press 'N'
    or 'S'. If a sticky tool activates, click on the canvas. Type. Escape.

EFFICIENCY:
  - You have ~7 turns. Each turn = one tool call (computer or sub_done).
  - You CAN combine multiple actions into one turn (the Anthropic API allows
    parallel tool calls). For example, in your last turn you can press Escape
    AND call sub_done at the same time.
  - As soon as you've TYPED the content, place the sticky is effectively done.
    Press Escape, then immediately call sub_done(success=true).

If anything obscures the canvas BEFORE you do anything else:
  - Onboarding modals: click Close / Skip / Got-it / Dismiss / "No thanks".
  - **Miro AI "Sidekick" sidebar** — a panel docked to the RIGHT edge of the
    canvas with the word "Sidekick" in its header, a dropdown chevron, and
    an X close button in the top-right corner of the panel. The panel shifts
    the entire canvas leftward; until it's closed your coord-based clicks
    will miss everything to the right of where you think you're clicking.
    To close it, do ANY of these (try in order):

      1. Click the SPARKLE / STAR icon at the top of the LEFT toolbar
         (small purple-and-blue gradient icon). This icon TOGGLES Sidekick —
         clicking it once while the sidebar is open will CLOSE the sidebar.
         This is usually the most reliable close path.
      2. Click the X button in the top-right corner of the Sidekick panel.
      3. Press the Escape key.

    After the sidebar is closed, the canvas snaps back to its full width and
    you can resume placement.

DO NOT click the sparkle/star icon when Sidekick is ALREADY CLOSED — that
would OPEN it (the icon toggles). The sparkle is also easy to mistake for
the sticky-note tool; the actual sticky-note tool is HIGHER up in the left
toolbar (a yellow square shape), not the sparkle at the bottom.

If after 5 actions you genuinely cannot place a sticky, call
sub_done(success=false, notes='<what blocked you>') so the system can move on."""


# ── Phase-0 analysis: agent inspects board, returns plan ───────────────────

ANALYZE_BOARD_TOOL = {
    "name": "analyze_board",
    "description": (
        "Report your assessment of the current Miro board and the plan of "
        "actions needed to bring it to the target state. Call exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "board_state": {
                "type": "string",
                "enum": ["empty", "incomplete", "unnecessary", "needs_cleaning", "complete"],
                "description": (
                    "Overall classification of the board:\n"
                    "  empty           — no relevant content placed yet.\n"
                    "  incomplete      — some target items present, others missing.\n"
                    "  unnecessary     — has off-topic / stale / duplicate items that shouldn't be there.\n"
                    "  needs_cleaning  — target items present but layout is overlapping / unreadable.\n"
                    "  complete        — all target items present, layout fine, nothing to do."
                ),
            },
            "observed_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["paper", "idea", "header", "other"]},
                        "content_excerpt": {
                            "type": "string",
                            "description": "First ~80 chars of the sticky/card text — used later to identify it for deletion.",
                        },
                        "matches_target": {
                            "type": "string",
                            "description": "Title (or 'P<i>'/'I<i>') of the target paper/idea this matches, or 'none' if it doesn't match anything in the target list.",
                        },
                    },
                    "required": ["kind", "content_excerpt", "matches_target"],
                },
                "description": "Inventory of each visible sticky/card on the board.",
            },
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["place_paper", "place_idea", "delete_item", "reorganize"],
                        },
                        "target_index": {
                            "type": "integer",
                            "description": "0-indexed position in the target papers list (place_paper) or target ideas list (place_idea). Required for place_*.",
                        },
                        "delete_target": {
                            "type": "string",
                            "description": "content_excerpt of the item to delete. Required for delete_item.",
                        },
                        "zone": {
                            "type": "string",
                            "description": "Optional placement hint (e.g. 'upper-left', 'right of existing items'). Used only for place_*.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "One-sentence justification for this action.",
                        },
                    },
                    "required": ["type", "reason"],
                },
                "description": "Ordered list of actions. Empty list means board is complete and nothing needs to happen.",
            },
            "summary": {"type": "string", "description": "1-2 sentence overall assessment."},
        },
        "required": ["board_state", "observed_items", "actions", "summary"],
    },
}


ANALYZE_SYSTEM = """You are auditing a Miro whiteboard before placing new content on it.

You see:
  - The CURRENT BOARD: a screenshot of what's already there.
  - The TARGET LIST: papers and ideas this board SHOULD contain.

Your job:
  1. Look carefully at the board. If items extend past the viewport, use the
     computer tool to scroll, zoom out (Cmd+- or Ctrl+-), or press Shift+1
     to fit-to-content. 1-2 navigation actions are usually enough; don't
     over-explore.
  2. Inventory each visible sticky / card. For each, decide whether it
     matches a target paper, a target idea, a section header, or nothing
     (clutter / off-topic).
  3. Classify the overall board_state as exactly one of:
        empty | incomplete | unnecessary | needs_cleaning | complete
  4. Decide actions:
        - place_paper / place_idea — for each TARGET item NOT yet on the
          board. Set target_index to the 0-indexed position in the target
          list (P0, P1, … for papers; I0, I1, … for ideas).
        - delete_item — for each existing item that doesn't belong
          (off-topic, duplicate, stale). Set delete_target to a recognisable
          excerpt (~30-60 chars) of its content so the next agent can find
          it.
        - reorganize — only if items overlap so badly the layout is unusable.
  5. Call analyze_board exactly once with the plan.

Constraints:
  - At most 7 actions.
  - Section headers like "📄 Papers" or "💡 Ideas" are part of the board's
    structure. Never delete them.
  - Don't over-clean: a few user-added notes that look harmless can stay.
  - If the board is already complete, return actions=[] and board_state=complete.

You MUST call analyze_board to exit. Do not answer in prose."""


ANALYZE_CLEANUP_SYSTEM = """You are doing a PRE-FLIGHT CLEANUP of a Miro whiteboard.

After you finish, REST will place any MISSING target papers (as colored
circles) and a CU pass will place any MISSING target ideas (as yellow
stickies). REST is IDEMPOTENT — items already on the board that match the
target list will be reused, NOT duplicated.

So your only job is to identify CLUTTER that should be REMOVED:
  - Items that DON'T match any paper or idea in the target list (off-topic
    stickies, items from a different topic, half-typed orphans, test
    placeholders).
  - Duplicates of one another.
  - Vestigial labels like "📄 Papers" or "💡 Ideas" — the new layout
    doesn't use section headers, so any such labels found should be
    deleted.

CRITICAL CONSTRAINTS — read carefully:
  - DO NOT emit delete_item for items that match a target paper or idea
    by content. A paper sticky/circle for "Attention Is All You Need"
    when "Attention Is All You Need" is in the target list is FINE —
    leave it. The next phase will recognise it via its state file and
    skip re-placement. Deleting it would create a stale-state bug where
    REST thinks the item exists but the board no longer has it.
  - DO NOT emit place_paper / place_idea / reorganize. The action list
    must contain only delete_item entries (or be empty).
  - If nothing needs cleaning, return actions=[].

Use the computer tool only if items are off-screen (scroll or Shift+1 to
fit-to-content). 1-2 navigation actions max — don't over-explore.

board_state classification (informational; doesn't drive behaviour):
    empty | incomplete | unnecessary | needs_cleaning | complete

You MUST call analyze_board to exit. Do not answer in prose."""


# ── Per-item delete sub-loop ───────────────────────────────────────────────

DELETE_DONE_TOOL = {
    "name": "delete_done",
    "description": "Call exactly once after deleting the requested sticky, or after concluding it cannot be found / deleted.",
    "input_schema": {
        "type": "object",
        "properties": {
            "success": {"type": "boolean", "description": "True if the sticky was deleted and is no longer visible."},
            "notes": {"type": "string", "description": "Brief note (e.g. 'deleted', 'sticky not found', 'modal blocked')."},
        },
        "required": ["success"],
    },
}


DELETE_SYSTEM = """You are removing ONE sticky/card from a Miro whiteboard.

The user message tells you what content to look for (a short excerpt). Your
job:
  1. Find that sticky on the canvas. If it's not visible, scroll or press
     Shift+1 to fit-to-content briefly (1-2 navigation actions max).
  2. Click it ONCE to select it (a selection outline appears).
  3. Press the Delete key (or Backspace). The sticky should disappear.
  4. Verify it's gone, then call delete_done(success=true).

If a Miro AI "Sidekick" sidebar appears on the right edge of the canvas
(header text "Sidekick", AI suggestions in the body), close it FIRST. The
most reliable close path: click the SPARKLE / STAR icon at the bottom of
the LEFT toolbar (purple-and-blue gradient icon). That icon TOGGLES
Sidekick — clicking it while the sidebar is open will close the sidebar.
You can also click the X in the panel's top-right or press Escape.

(Don't click the sparkle when Sidekick is already CLOSED — it would re-open
the sidebar.)

If after 3-4 actions you genuinely cannot find or delete the sticky, call
delete_done(success=false, notes='<what went wrong>') so the system moves on.

CRITICAL: only delete the sticky whose content matches the request. Do NOT
delete section headers ("📄 Papers", "💡 Ideas") even if they're nearby.

You MUST call delete_done to exit. Do not answer in prose."""


# ─────────────────────────────────────────────────────────────────────────────
# Per-item sub-loops
# ─────────────────────────────────────────────────────────────────────────────

def place_one_sticky_cu(
    client,
    page,
    content: str,
    kind: str,
    zone_hint: str,
    prior_placements: list[tuple[int, int]],
    debug_label: str = "",
) -> tuple[bool, int | None, int | None]:
    """One CU sub-loop to place a single sticky. Returns (success, approx_x, approx_y).

    Each call has a fresh, narrow context — no history bloat from previous items.
    Bounded to MIRO_CU_PER_ITEM_STEPS turns. The agent is told approximate
    positions of items already placed so it can avoid overlapping them.

    Debug output: each turn, prints the agent's text reasoning + each CU
    action it requests. After the sub-loop, saves the final screenshot to
    /tmp/cu_<debug_label>_final.png so we can see what state the board ended in.
    """
    # Start each sub-loop with a clean canvas: deselect any item AND close
    # any Miro overlay (the AI Sidekick panel especially — Miro pops it up
    # when it detects automation patterns and the right-side sidebar shifts
    # the canvas, breaking coord-based clicks).
    close_sidekick_panel(page)
    initial_png = page.screenshot(full_page=False)
    cu_tool = cu_tool_spec(width=MIRO_CU_VIEWPORT_W, height=MIRO_CU_VIEWPORT_H)

    prior_text = ""
    if prior_placements:
        listing = "\n".join(f"    - around ({x}, {y})" for x, y in prior_placements if x is not None)
        if listing:
            prior_text = f"Already-placed stickies (avoid overlapping these viewport positions):\n{listing}\n\n"

    user_prompt = (
        f"Place ONE {kind} sticky note now.\n\n"
        f"Content (type this verbatim):\n  {content}\n\n"
        f"Zone: {zone_hint}.\n\n"
        f"{prior_text}"
        f"Use whichever method (A/B/C from the system prompt) works. Take the action,\n"
        f"verify in the next screenshot, then call sub_done."
    )

    # Build the user content: instructions → current board screenshot.
    user_content: list[dict] = [
        {"type": "text", "text": user_prompt},
        {"type": "text", "text": "CURRENT BOARD STATE (what you actually see right now):"},
        as_image_block(initial_png),
    ]

    messages: list[dict] = [{"role": "user", "content": user_content}]

    last_png: bytes | None = initial_png
    for step in range(MIRO_CU_PER_ITEM_STEPS):
        final = step >= MIRO_CU_PER_ITEM_STEPS - 1
        tools = [SUB_DONE_TOOL] if final else [cu_tool, SUB_DONE_TOOL]

        try:
            resp = client.beta.messages.create(
                model=MIRO_CU_MODEL,
                max_tokens=500,
                system=[{"type": "text", "text": SUB_LOOP_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                tools=tools,
                messages=messages,
                betas=[COMPUTER_USE_BETA],
            )
        except Exception as e:
            print(f"      sub-loop API error: {type(e).__name__}: {e}")
            return False, None, None

        # Debug: print agent's text reasoning + each tool_use
        for block in resp.content:
            if block.type == "text" and block.text.strip():
                first_line = block.text.strip().split("\n")[0][:140]
                print(f"      [t{step + 1}] reasoning: {first_line}")
            elif block.type == "tool_use":
                if block.name == "computer":
                    action = block.input.get("action", "?")
                    detail = ""
                    if action == "key":
                        detail = f" '{block.input.get('text', '')}'"
                    elif action in ("left_click", "double_click", "mouse_move"):
                        detail = f" @ {block.input.get('coordinate')}"
                    elif action == "type":
                        detail = f" {block.input.get('text', '')[:40]!r}"
                    elif action == "scroll":
                        detail = f" {block.input.get('scroll_direction')} {block.input.get('scroll_amount')}"
                    print(f"      [t{step + 1}] action: {action}{detail}")
                elif block.name == "sub_done":
                    inp = block.input
                    print(f"      [t{step + 1}] sub_done: success={inp.get('success')} notes={inp.get('notes', '')!r}")

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            print(f"      [t{step + 1}] no tool use — agent stopped")
            return False, None, None

        messages.append({"role": "assistant", "content": resp.content})

        sub_done_input: dict | None = None
        tool_results: list[dict] = []
        for block in tool_uses:
            if block.name == "sub_done":
                sub_done_input = block.input
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Acknowledged. Sub-session ending.",
                })
            elif block.name == "computer":
                png, is_error = execute_cu_action(page, block.input)
                last_png = png
                # Anthropic's API rejects tool_results with `is_error: true`
                # plus image content. Prepend a text note instead and keep the
                # screenshot so the agent can see the (likely unchanged) state.
                content_blocks: list[dict] = []
                if is_error:
                    content_blocks.append({"type": "text", "text": "(previous action returned an error; the screenshot below shows the current state)"})
                content_blocks.append(as_image_block(png))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content_blocks,
                })

        messages.append({"role": "user", "content": tool_results})

        if sub_done_input is not None:
            # Debug: dump final screenshot for inspection.
            if debug_label and last_png:
                Path(f"/tmp/cu_{debug_label}_final.png").write_bytes(last_png)
            return (
                bool(sub_done_input.get("success", False)),
                sub_done_input.get("approx_x"),
                sub_done_input.get("approx_y"),
            )

    if debug_label and last_png:
        Path(f"/tmp/cu_{debug_label}_final.png").write_bytes(last_png)
    return False, None, None


def analyze_board_cu(
    client,
    page,
    target_papers: list[dict],
    target_ideas: list[dict],
    max_steps: int = 4,
    mode: str = "full",
) -> dict | None:
    """Phase-0 sub-loop: agent inspects the current board and returns a plan dict.

    The plan has shape:
        {board_state, observed_items, actions, summary}
    where actions is the ordered list the executor consumes.

    `mode` selects the system prompt:
      - "full"    — agent may emit place_paper / place_idea / delete_item / reorganize.
      - "cleanup" — pre-flight pass that may emit ONLY delete_item actions.
                    Used by the new flow where REST handles paper placement and
                    a separate CU pass places idea stickies.

    Returns None if the agent fails to call analyze_board within max_steps,
    in which case the caller falls back to direct placement.
    """
    close_sidekick_panel(page)
    initial_png = page.screenshot(full_page=False)
    cu_tool = cu_tool_spec(width=MIRO_CU_VIEWPORT_W, height=MIRO_CU_VIEWPORT_H)

    target_lines: list[str] = []
    if target_papers:
        target_lines.append("TARGET PAPERS (will be placed by REST after cleanup):" if mode == "cleanup" else "TARGET PAPERS (board should contain these):")
        for i, p in enumerate(target_papers):
            rating = p.get("review", {}).get("rating", "?")
            target_lines.append(f"  [P{i}] {p['title'][:90]}  (rating {rating}/10)")
    if target_ideas:
        target_lines.append("")
        target_lines.append("TARGET IDEAS (will be placed by a later CU pass):" if mode == "cleanup" else "TARGET IDEAS (board should contain these):")
        for i, idea in enumerate(target_ideas):
            target_lines.append(f"  [I{i}] {idea['title'][:90]}")
    target_text = "\n".join(target_lines) if target_lines else "(no target items.)"

    closer = (
        "Identify items to delete (if any) and call analyze_board with delete_item actions only."
        if mode == "cleanup"
        else "Inventory the board, classify it, and call analyze_board with the plan."
    )
    user_content: list[dict] = [
        {"type": "text", "text": target_text},
        {"type": "text", "text": "CURRENT BOARD (what the saved Miro session shows right now):"},
        as_image_block(initial_png),
        {"type": "text", "text": closer},
    ]
    messages: list[dict] = [{"role": "user", "content": user_content}]

    system_prompt = ANALYZE_CLEANUP_SYSTEM if mode == "cleanup" else ANALYZE_SYSTEM

    for step in range(max_steps):
        final = step >= max_steps - 1
        tools = [ANALYZE_BOARD_TOOL] if final else [cu_tool, ANALYZE_BOARD_TOOL]

        try:
            resp = client.beta.messages.create(
                model=MIRO_CU_MODEL,
                max_tokens=2000,
                system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                tools=tools,
                messages=messages,
                betas=[COMPUTER_USE_BETA],
            )
        except Exception as e:
            print(f"      [analyze] API error: {type(e).__name__}: {e}")
            return None

        for block in resp.content:
            if block.type == "text" and block.text.strip():
                first_line = block.text.strip().splitlines()[0][:140]
                print(f"      [analyze t{step + 1}] reasoning: {first_line}")
            elif block.type == "tool_use":
                if block.name == "computer":
                    print(f"      [analyze t{step + 1}] action: {block.input.get('action', '?')}")
                elif block.name == "analyze_board":
                    print(f"      [analyze t{step + 1}] analyze_board called")

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            print(f"      [analyze t{step + 1}] no tool use — agent stopped without a plan")
            return None

        messages.append({"role": "assistant", "content": resp.content})

        plan: dict | None = None
        tool_results: list[dict] = []
        for block in tool_uses:
            if block.name == "analyze_board":
                plan = block.input
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Plan received. Sub-session ending.",
                })
            elif block.name == "computer":
                png, is_error = execute_cu_action(page, block.input)
                content_blocks: list[dict] = []
                if is_error:
                    content_blocks.append({"type": "text", "text": "(previous action returned an error; the screenshot below shows the current state)"})
                content_blocks.append(as_image_block(png))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content_blocks,
                })

        messages.append({"role": "user", "content": tool_results})

        if plan is not None:
            return plan

    return None


def delete_one_sticky_cu(
    client,
    page,
    content_excerpt: str,
    max_steps: int = 5,
    debug_label: str = "",
) -> bool:
    """CU sub-loop: find the sticky whose text matches content_excerpt and delete it.

    Returns True if the agent confirms the sticky is gone.
    """
    close_sidekick_panel(page)
    initial_png = page.screenshot(full_page=False)
    cu_tool = cu_tool_spec(width=MIRO_CU_VIEWPORT_W, height=MIRO_CU_VIEWPORT_H)

    user_content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Delete the sticky/card whose content matches this excerpt:\n\n"
                f"  {content_excerpt[:200]}\n\n"
                f"Find it on the canvas, click once to select it, press the three dots and press Delete, "
                f"verify it's gone, then call delete_done."
            ),
        },
        as_image_block(initial_png),
    ]
    messages: list[dict] = [{"role": "user", "content": user_content}]

    last_png: bytes | None = initial_png
    for step in range(max_steps):
        final = step >= max_steps - 1
        tools = [DELETE_DONE_TOOL] if final else [cu_tool, DELETE_DONE_TOOL]

        try:
            resp = client.beta.messages.create(
                model=MIRO_CU_MODEL,
                max_tokens=500,
                system=[{"type": "text", "text": DELETE_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                tools=tools,
                messages=messages,
                betas=[COMPUTER_USE_BETA],
            )
        except Exception as e:
            print(f"      [delete] API error: {type(e).__name__}: {e}")
            return False

        for block in resp.content:
            if block.type == "text" and block.text.strip():
                first_line = block.text.strip().splitlines()[0][:140]
                print(f"      [delete t{step + 1}] reasoning: {first_line}")
            elif block.type == "tool_use":
                if block.name == "computer":
                    action = block.input.get("action", "?")
                    detail = ""
                    if action == "key":
                        detail = f" '{block.input.get('text', '')}'"
                    elif action in ("left_click", "double_click", "mouse_move"):
                        detail = f" @ {block.input.get('coordinate')}"
                    print(f"      [delete t{step + 1}] action: {action}{detail}")
                elif block.name == "delete_done":
                    inp = block.input
                    print(f"      [delete t{step + 1}] delete_done: success={inp.get('success')} notes={inp.get('notes', '')!r}")

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            return False

        messages.append({"role": "assistant", "content": resp.content})

        done_input: dict | None = None
        tool_results: list[dict] = []
        for block in tool_uses:
            if block.name == "delete_done":
                done_input = block.input
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": "Acknowledged."})
            elif block.name == "computer":
                png, is_error = execute_cu_action(page, block.input)
                last_png = png
                # Anthropic's API rejects tool_results with `is_error: true`
                # plus image content. Prepend a text note instead and keep the
                # screenshot so the agent can see the (likely unchanged) state.
                content_blocks: list[dict] = []
                if is_error:
                    content_blocks.append({"type": "text", "text": "(previous action returned an error; the screenshot below shows the current state)"})
                content_blocks.append(as_image_block(png))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content_blocks,
                })

        messages.append({"role": "user", "content": tool_results})

        if done_input is not None:
            if debug_label and last_png:
                Path(f"/tmp/cu_{debug_label}_final.png").write_bytes(last_png)
            return bool(done_input.get("success", False))

    if debug_label and last_png:
        Path(f"/tmp/cu_{debug_label}_final.png").write_bytes(last_png)
    return False


def execute_plan_cu(
    client,
    page,
    plan: dict,
    target_papers: list[dict],
    target_ideas: list[dict],
) -> tuple[list[dict], list[dict], int]:
    """Iterate over plan['actions'] and dispatch each to the right CU sub-loop.

    Returns (placed_papers, placed_ideas, deletions_count). place_* actions reuse
    `place_one_sticky_cu`; delete_item uses `delete_one_sticky_cu`; reorganize
    is logged but skipped for now.
    """
    placed_papers: list[dict] = []
    placed_ideas: list[dict] = []
    deletions = 0
    prior_placements: list[tuple[int, int]] = []

    actions = plan.get("actions", []) or []
    print(f"\n  [analysis] board_state={plan.get('board_state')}; {len(actions)} action(s) planned")
    if plan.get("summary"):
        print(f"  [analysis] {plan['summary']}")

    for i, action in enumerate(actions, 1):
        atype = action.get("type")
        reason = (action.get("reason") or "").strip()
        zone = (action.get("zone") or "").strip()

        if atype == "place_paper":
            idx = action.get("target_index")
            if idx is None or not (0 <= idx < len(target_papers)):
                print(f"  [act {i}] place_paper: bad target_index {idx} — skip")
                continue
            p = target_papers[idx]
            content = f"📄 {p['title'][:90]}\n(rating {p['review']['rating']}/10)"
            zone_hint = zone or "UPPER HALF of the visible canvas (near the '📄 Papers' header)"
            print(f"  [act {i}] place_paper P{idx}: {p['title'][:50]}  ({reason})")
            ok, x, y = place_one_sticky_cu(
                client, page, content,
                kind="paper",
                zone_hint=zone_hint,
                prior_placements=prior_placements,
                debug_label=f"plan_paper{idx}",
            )
            if ok:
                placed_papers.append(p)
                if x is not None and y is not None:
                    prior_placements.append((x, y))
            page.wait_for_timeout(MIRO_CU_PACE_SECONDS * 1000)

        elif atype == "place_idea":
            idx = action.get("target_index")
            if idx is None or not (0 <= idx < len(target_ideas)):
                print(f"  [act {i}] place_idea: bad target_index {idx} — skip")
                continue
            idea = target_ideas[idx]
            rating = idea.get("novelty_rating")
            rating_line = f"(novelty {rating}/10)\n" if rating else ""
            content = f"💡 {idea['title'][:90]}\n{rating_line}{idea.get('problem', '')[:140]}"
            zone_hint = zone or "LOWER HALF of the visible canvas (near the '💡 Ideas' header)"
            print(f"  [act {i}] place_idea I{idx}: {idea['title'][:50]}  ({reason})")
            ok, x, y = place_one_sticky_cu(
                client, page, content,
                kind="idea",
                zone_hint=zone_hint,
                prior_placements=prior_placements,
                debug_label=f"plan_idea{idx}",
            )
            if ok:
                placed_ideas.append(idea)
                if x is not None and y is not None:
                    prior_placements.append((x, y))
            page.wait_for_timeout(MIRO_CU_PACE_SECONDS * 1000)

        elif atype == "delete_item":
            target = (action.get("delete_target") or "").strip()
            if not target:
                print(f"  [act {i}] delete_item: no delete_target — skip")
                continue
            print(f"  [act {i}] delete_item: {target[:60]!r}  ({reason})")
            if delete_one_sticky_cu(client, page, target, debug_label=f"plan_del{i}"):
                deletions += 1
            page.wait_for_timeout(MIRO_CU_PACE_SECONDS * 1000)

        elif atype == "reorganize":
            print(f"  [act {i}] reorganize: not yet implemented — skipped  ({reason})")

        else:
            print(f"  [act {i}] unknown action type {atype!r} — skip")

    return placed_papers, placed_ideas, deletions


# ─────────────────────────────────────────────────────────────────────────────
# Top-level CU backend
# ─────────────────────────────────────────────────────────────────────────────

def post_topic_via_cu(topic: str, papers: list[dict], ideas: list[dict]) -> dict:
    """Layered flow — REST builds structure, CU places ideas, REST connects.

    Phases:
      0. SETUP                — REST: ensure board, pre-stage section headers.
      1. PRE-FLIGHT CLEANUP   — open browser; CU runs `analyze_board_cu` in
                                cleanup mode and `execute_plan_cu` deletes
                                stale stickies (delete_item actions only;
                                place_* and reorganize are filtered out).
      2. PAPER NODES (REST)   — `post_paper_circle_rest` places one colored circle per
                                paper at known positions. Capture item IDs
                                and positions so connectors can be drawn
                                later.
      3. IDEA STICKIES (CU)   — for each idea, run `place_one_sticky_cu`
                                with a zone hint that names the idea's
                                source-paper titles ("between the cards
                                titled X, Y, Z"). The agent visually targets
                                the convergence point of those papers.
      4. CONNECTORS (REST)    — `match_idea_stickies_rest` queries the
                                board, matches new stickies to ideas by
                                title-prefix, then `connect_rest` draws one
                                connector per (source_paper → idea_sticky).

    State is updated for both REST-placed papers (real Miro item IDs) and
    CU-placed ideas (matched item IDs). Browser-level exceptions still fall
    back to the deterministic REST backend when MIRO_FALLBACK_TO_REST=1.
    """
    if not MIRO_SESSION_FILE.exists():
        msg = f"No saved Miro session at {MIRO_SESSION_FILE}. Run setup_miro_login.py first."
        if MIRO_FALLBACK_TO_REST:
            print(f"  ! {msg} — falling back to REST")
            return {"fell_back": True, **post_topic_via_rest(topic, papers, ideas)}
        raise RuntimeError(msg)

    # ── Phase 0: setup ─────────────────────────────────────────────────────
    board_id, view_url = ensure_board_for_topic(topic)

    client = make_client()
    plan: dict | None = None
    deletions = 0
    paper_id_by_index: dict[int, str] = {}
    placed_idea_indices: list[int] = []
    idea_id_by_index: dict[int, str] = {}
    n_connectors = 0
    notes = ""

    try:
        with shared_or_fresh_page(
            viewport={"width": MIRO_CU_VIEWPORT_W, "height": MIRO_CU_VIEWPORT_H},
            storage_state=str(MIRO_SESSION_FILE),
        ) as page:
            page.goto(view_url, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_timeout(5000)
            close_sidekick_panel(page)
            page.wait_for_timeout(500)

            # ── Phase 1: pre-flight cleanup ─────────────────────────────────
            print("\n  [Miro CU] phase 1: pre-flight cleanup analysis…")
            plan = analyze_board_cu(client, page, papers, ideas, mode="cleanup")
            if plan is None:
                print("  [Miro CU] cleanup analysis returned no plan — skipping deletions.")
            else:
                # Defensive filter: cleanup mode should only emit delete_item,
                # but if the agent slips a place_* in, drop it before executing.
                cleanup_actions = [a for a in (plan.get("actions") or []) if a.get("type") == "delete_item"]
                if cleanup_actions:
                    plan["actions"] = cleanup_actions
                    _, _, deletions = execute_plan_cu(client, page, plan, papers, ideas)
                else:
                    print(f"  [Miro CU] cleanup: nothing to delete (board_state={plan.get('board_state')!r}).")

            # ── Phase 2: REST places paper circles via NetworkX layout ─────
            # Spring layout over the (paper ↔ idea) edge graph gives positions
            # that already place ideas between their source papers — even though
            # CU will place the actual idea stickies, the paper layout is graph-
            # aware so the agent's vision-based placement converges naturally.
            print(f"\n  [Miro CU] phase 2: REST placing {len(papers)} paper circle(s) via spring layout…")
            paper_positions, idea_positions = compute_layout(papers, ideas)
            for i, p in enumerate(papers):
                x, y = paper_positions.get(i, (0.0, 0.0))
                try:
                    pid = post_paper_circle_rest(board_id, p, (x, y), color_index=i)
                    paper_id_by_index[i] = pid
                except requests.HTTPError as e:
                    print(f"    ! paper circle P{i} failed: {e}")

            # Bring the just-placed papers into the agent's viewport.
            # The pre-zoom only works if the canvas has focus, otherwise the
            # keyboard shortcuts go nowhere. We click a small offset from the
            # top-left corner of the viewport (200, 200) — far from the spring
            # layout's central nodes (origin ± 700 board units) so the click
            # lands on empty canvas, not on a circle. Escape after the click
            # deselects in case Miro's auto-pan repositions a circle there.
            try:
                page.mouse.click(200, 200)
                page.wait_for_timeout(200)
                page.keyboard.press("Escape")
                page.wait_for_timeout(150)
                page.keyboard.press("Shift+1")        # Miro: fit content
                page.wait_for_timeout(800)
                for _ in range(3):
                    page.keyboard.press("Control+Equal")  # Miro: zoom in
                    page.wait_for_timeout(150)
                page.keyboard.press("Escape")
            except Exception:
                pass
            page.wait_for_timeout(1000)

            # ── Phase 3: CU places idea stickies at edge convergence points ─
            if not ideas:
                print("\n  [Miro CU] phase 3: no ideas to place; skipping CU pass.")
            else:
                print(f"\n  [Miro CU] phase 3: CU placing {len(ideas)} idea sticky(s) at convergence points…")

            # Pre-skip ideas already on the board (e.g. preserved by cleanup
            # because they matched the target). Query stickies via REST, match
            # by title-prefix, record their IDs so phase 4 can connect them.
            already_on_board = match_idea_stickies_rest(board_id, ideas)
            for j, sid in already_on_board.items():
                idea_id_by_index[j] = sid
                placed_idea_indices.append(j)
                print(f"    [idea I{j}] already on board (matched sticky {sid}); skipping CU placement.")

            prior_placements: list[tuple[int, int]] = []
            for i, idea in enumerate(ideas):
                if i in already_on_board:
                    continue
                src_indices = [
                    si for si in (idea.get("source_paper_indices") or [])
                    if 0 <= si < len(papers)
                ]
                if len(src_indices) < 2:
                    print(f"    [idea I{i}] only {len(src_indices)} valid source(s); skipping CU placement.")
                    continue

                source_titles = [papers[si]["title"][:50] for si in src_indices]
                source_desc = ", ".join(f"'{t}'" for t in source_titles)
                rating = idea.get("novelty_rating")
                rating_line = f"(novelty {rating}/10)\n" if rating else ""
                content = f"💡 {idea['title'][:90]}\n{rating_line}{idea.get('problem', '')[:140]}"
                zone_hint = (
                    f"BETWEEN the colored paper circles titled {source_desc}. "
                    f"Place the sticky at the visual midpoint of those circles "
                    f"(connectors from each source circle will be drawn to this sticky, "
                    f"so positioning at the centroid makes the edges visually converge here)."
                )
                print(f"    [idea I{i}] CU placing: {idea['title'][:50]}  (sources: {src_indices})")
                ok, x, y = place_one_sticky_cu(
                    client, page, content,
                    kind="idea",
                    zone_hint=zone_hint,
                    prior_placements=prior_placements,
                    debug_label=f"idea{i}",
                )
                if ok:
                    placed_idea_indices.append(i)
                    if x is not None and y is not None:
                        prior_placements.append((x, y))
                else:
                    # CU couldn't place this sticky — fall back to REST at the
                    # spring-layout's precomputed convergence point so the run
                    # still produces a complete board. Connectors in phase 4
                    # will pick up the REST-placed sticky via content match.
                    fx, fy = idea_positions.get(i, (0.0, 0.0))
                    try:
                        iid = post_idea_sticky_rest(board_id, idea, (fx, fy))
                        idea_id_by_index[i] = iid
                        placed_idea_indices.append(i)
                        print(f"    [idea I{i}] CU failed — REST placed at ({fx:.0f}, {fy:.0f}) as item {iid}")
                    except requests.HTTPError as e:
                        print(f"    [idea I{i}] CU failed AND REST fallback failed: {e}")
                page.wait_for_timeout(MIRO_CU_PACE_SECONDS * 1000)

    except Exception as e:
        notes = f"CU exception: {type(e).__name__}: {e}"
        if MIRO_FALLBACK_TO_REST:
            print(f"  ! Miro CU agent failed — falling back to REST. ({notes})")
            return {"fell_back": True, **post_topic_via_rest(topic, papers, ideas)}
        raise

    # ── Phase 4: REST queries new stickies, draws connectors ──────────────
    n_skipped = 0
    if placed_idea_indices:
        placed_ideas = [ideas[i] for i in placed_idea_indices]
        print(f"\n  [Miro CU] phase 4: REST matching {len(placed_ideas)} CU-placed sticky(s) and drawing connectors…")
        matched = match_idea_stickies_rest(board_id, placed_ideas)
        # Re-key matched dict from "position-in-placed_ideas" → "absolute idea index".
        for local_idx, sid in matched.items():
            if 0 <= local_idx < len(placed_idea_indices):
                idea_id_by_index[placed_idea_indices[local_idx]] = sid

        # Dedupe check: pull existing (start, end) pairs once so we don't create
        # duplicate connectors on re-runs (Miro doesn't dedupe; each call layers
        # another connector on top of the prior one).
        existing_pairs = fetch_existing_connector_pairs(board_id)

        for i in placed_idea_indices:
            iid = idea_id_by_index.get(i)
            if not iid:
                print(f"    [idea I{i}] could not match sticky on board — skipping connectors.")
                continue
            for src_idx in (ideas[i].get("source_paper_indices") or []):
                pid = paper_id_by_index.get(src_idx)
                if not pid:
                    continue
                if (pid, iid) in existing_pairs:
                    n_skipped += 1
                    continue
                if connect_rest(board_id, pid, iid):
                    n_connectors += 1
                    existing_pairs.add((pid, iid))
                    existing_pairs.add((iid, pid))
        print(f"  [Miro CU] phase 4: drew {n_connectors} connector(s); skipped {n_skipped} (already present).")
    else:
        print("\n  [Miro CU] phase 4: no CU-placed ideas to connect; skipping.")

    # ── State updates ──────────────────────────────────────────────────────
    state = load_state()
    items = items_for_board(state, board_id)
    for i, p in enumerate(papers):
        if i in paper_id_by_index:
            items["papers"][p["url"]] = paper_id_by_index[i]
    for i in placed_idea_indices:
        h = idea_hash(ideas[i])
        items["ideas"][h] = idea_id_by_index.get(i, "cu_placed")
    save_state(state)

    return {
        "backend": "cu",
        "board_id": board_id,
        "view_url": view_url,
        "plan": plan,
        "papers_placed": len(paper_id_by_index),
        "ideas_placed": len(placed_idea_indices),
        "deletions": deletions,
        "connectors_drawn": n_connectors,
        "connectors_skipped": n_skipped,
        "items_placed": len(paper_id_by_index) + len(placed_idea_indices),
        "items_attempted": len(papers) + len(ideas),
        "success": (len(paper_id_by_index) + len(placed_idea_indices) + deletions) > 0,
        "notes": notes,
    }
