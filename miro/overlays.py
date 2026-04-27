"""Close Miro's AI 'Sidekick' panel if it pops up during a CU run.

Miro detects automation patterns and pops a docked right-edge sidebar
labeled 'Sidekick'. The panel shifts the canvas leftward, breaking every
coordinate-based click until closed. The close button's aria-label varies,
so we locate the panel by its visible header text and click the X position
at the viewport's right edge.

This is the only overlay we handle defensively — onboarding modals,
generic dialogs, and the Escape-key fallback were dropped because the
real failure mode is the CU agent clicking the wrong thing and triggering
Sidekick. One targeted close is enough; spraying generic dismisses just
masks coordinate bugs in the agent.
"""

from __future__ import annotations


def close_sidekick_panel(page) -> bool:
    """Close Miro's Sidekick panel if visible. Returns True iff a close
    click was issued."""
    try:
        sidekick = page.get_by_text("Sidekick", exact=False).first
        if not sidekick.is_visible(timeout=300):
            return False
        box = sidekick.bounding_box()
        if not box:
            return False
        # The X close button sits at the far right of the same row as the
        # Sidekick header. Click ~28 px in from the viewport's right edge
        # at the header's vertical center.
        vp = page.viewport_size or {"width": 1280, "height": 1600}
        click_x = vp["width"] - 28
        click_y = int(box["y"] + box["height"] / 2)
        page.mouse.click(click_x, click_y)
        page.wait_for_timeout(300)
        return True
    except Exception:
        return False
