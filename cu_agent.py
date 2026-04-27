"""Anthropic computer-use action handlers — the central CU substrate.

Everything related to executing computer-use tool actions on a Playwright page
lives here. Other modules (reviewer, miro) declare their own goal-tools
(submit_review, sub_done) and run their own agent loops, but the CU action
dispatch is shared.

What this file owns:
  - The computer_20251124 tool-spec constants and beta header.
  - xdotool → Playwright key-name translation.
  - execute_cu_action: dispatches each CU action type to the right Playwright
    primitive, returns (screenshot_bytes, is_error).
  - budget_hint: soft pressure on agent loops to terminate within budget.
  - cu_tool_spec(): factory for the tool definition dict used in messages.create.

What this file does NOT own:
  - Specific agent loops (those live with their goals).
  - Model selection / pricing / general Anthropic plumbing (helpers.py).
"""

from __future__ import annotations

import os

from playwright.sync_api import Page

from helpers import warn_exception

# ── CU tool spec (Anthropic's computer_20251124, used by Opus 4.7 / Sonnet 4.6 / Opus 4.6) ──

COMPUTER_USE_TOOL_TYPE = os.environ.get("CU_TOOL_TYPE", "computer_20251124")
COMPUTER_USE_BETA = os.environ.get("CU_BETA", "computer-use-2025-11-24")
CU_DISPLAY_WIDTH = int(os.environ.get("CU_DISPLAY_WIDTH", "1280"))
CU_DISPLAY_HEIGHT = int(os.environ.get("CU_DISPLAY_HEIGHT", "1600"))
AGENT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "16"))


def cu_tool_spec(width: int = CU_DISPLAY_WIDTH, height: int = CU_DISPLAY_HEIGHT, enable_zoom: bool = True) -> dict:
    """Build the tool definition for messages.create()."""
    return {
        "type": COMPUTER_USE_TOOL_TYPE,
        "name": "computer",
        "display_width_px": width,
        "display_height_px": height,
        "display_number": 1,
        "enable_zoom": enable_zoom,
    }


# ── Key translation (xdotool names → Playwright names) ─────────────────────

_XDOTOOL_KEY_MAP = {
    "Page_Down": "PageDown", "Page_Up": "PageUp",
    "Return": "Enter", "BackSpace": "Backspace",
    "Escape": "Escape", "Tab": "Tab",
    "Home": "Home", "End": "End",
    "Up": "ArrowUp", "Down": "ArrowDown",
    "Left": "ArrowLeft", "Right": "ArrowRight",
    "space": " ",
}
_MODIFIER_MAP = {"ctrl": "Control", "alt": "Alt", "shift": "Shift", "super": "Meta", "cmd": "Meta"}


def xdotool_to_playwright_key(text: str) -> str:
    """Map xdotool-style key names (e.g. 'ctrl+End', 'Page_Down') to Playwright."""
    parts = text.split("+")
    out = []
    for p in parts:
        out.append(_MODIFIER_MAP.get(p.lower(), _XDOTOOL_KEY_MAP.get(p, p)))
    return "+".join(out)


# ── Budget pressure helper (soft termination signal for agent loops) ───────

def budget_hint(step: int, max_steps: int) -> str | None:
    """Soft pressure on the agent to terminate within budget. Returns a text hint
    to inject into the next user message, or None if we're early in the loop."""
    remaining = max_steps - step - 1
    if remaining <= 0:
        return f"FINAL TURN ({step + 1}/{max_steps}). You must finalize now with the structured output tool."
    if remaining <= 2:
        return f"Turn {step + 1}/{max_steps}. Only {remaining} turn(s) left — wrap up and submit."
    if step >= max_steps // 2:
        return f"Turn {step + 1}/{max_steps}. You should have enough to form a verdict; finalize when ready."
    return None


# ── CU action dispatcher (the heart of this module) ────────────────────────

def execute_cu_action(page: Page, ipt: dict) -> tuple[bytes, bool]:
    """Execute a computer_20251124 tool action and return (screenshot_png, is_error).

    The tool_result for every CU action should include a fresh screenshot so
    Claude can see the new state — that's the canonical CU pattern.

    Supported actions (per Anthropic's docs for computer_20251124):
      screenshot, left_click, right_click, middle_click, double_click,
      triple_click, left_click_drag, left_mouse_down, left_mouse_up, type, key,
      hold_key, mouse_move, cursor_position, scroll, wait, zoom.

    For modifier keys passed via the `text` parameter on click/scroll, we wrap
    the action in keyboard.down/up around the primitive.
    """
    action = ipt["action"]
    try:
        if action == "screenshot":
            pass  # just return the current screenshot

        elif action == "left_click":
            x, y = ipt["coordinate"]
            modifier = ipt.get("text")
            if modifier:
                page.keyboard.down(_MODIFIER_MAP.get(modifier.lower(), modifier))
            page.mouse.click(x, y)
            if modifier:
                page.keyboard.up(_MODIFIER_MAP.get(modifier.lower(), modifier))

        elif action == "right_click":
            page.mouse.click(*ipt["coordinate"], button="right")
        elif action == "middle_click":
            page.mouse.click(*ipt["coordinate"], button="middle")
        elif action == "double_click":
            page.mouse.dblclick(*ipt["coordinate"])
        elif action == "triple_click":
            page.mouse.click(*ipt["coordinate"], click_count=3)

        elif action == "left_click_drag":
            x1, y1 = ipt["start_coordinate"]
            x2, y2 = ipt["coordinate"]
            page.mouse.move(x1, y1)
            page.mouse.down()
            page.mouse.move(x2, y2)
            page.mouse.up()

        elif action == "left_mouse_down":
            page.mouse.down()
        elif action == "left_mouse_up":
            page.mouse.up()

        elif action == "type":
            page.keyboard.type(ipt["text"])
        elif action == "key":
            page.keyboard.press(xdotool_to_playwright_key(ipt["text"]))
        elif action == "hold_key":
            key = xdotool_to_playwright_key(ipt["text"])
            duration = float(ipt.get("duration", 1))
            page.keyboard.down(key)
            page.wait_for_timeout(int(duration * 1000))
            page.keyboard.up(key)

        elif action == "mouse_move":
            page.mouse.move(*ipt["coordinate"])
        elif action == "cursor_position":
            pass  # we don't track; just return current screenshot

        elif action == "scroll":
            x, y = ipt["coordinate"]
            direction = ipt.get("scroll_direction", "down")
            amount = int(ipt.get("scroll_amount", 3))
            page.mouse.move(x, y)
            dy = amount * 200 * (1 if direction == "down" else -1) if direction in ("down", "up") else 0
            dx = amount * 200 * (1 if direction == "right" else -1) if direction in ("left", "right") else 0
            modifier = ipt.get("text")
            if modifier:
                page.keyboard.down(_MODIFIER_MAP.get(modifier.lower(), modifier))
            page.mouse.wheel(dx, dy)
            if modifier:
                page.keyboard.up(_MODIFIER_MAP.get(modifier.lower(), modifier))

        elif action == "wait":
            page.wait_for_timeout(int(float(ipt.get("duration", 1)) * 1000))

        elif action == "zoom":
            # Return a screenshot of just the requested region — Claude looks
            # at it at full resolution, which is the whole point of zoom.
            x1, y1, x2, y2 = ipt["region"]
            return page.screenshot(clip={"x": x1, "y": y1, "width": max(1, x2 - x1), "height": max(1, y2 - y1)}), False

        else:
            return page.screenshot(full_page=False), True  # unknown action → still return screenshot, mark as error

    except Exception as e:
        warn_exception(f"CU action {action!r}", e)
        return page.screenshot(full_page=False), True

    page.wait_for_timeout(150)
    return page.screenshot(full_page=False), False
