"""One-time setup: log in to Miro manually, save the session for future runs.

main.py calls `run_miro_login()` automatically on first start when the
session file is missing — so a fresh user typically never runs this script
directly. It still works as a standalone tool when you need to refresh an
expired session:

    python setup_miro_login.py

It opens a Chromium window at the Miro dashboard; you log in normally; you
press Enter in the terminal. Playwright saves cookies + local-storage to
.miro_session.json, and subsequent CU runs load that file via storage_state=
so the agent itself never sees credentials.

Re-run if your Miro session expires (cookies expire after some weeks).
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

STORAGE_STATE = Path(__file__).parent / ".miro_session.json"
DASHBOARD_URL = "https://miro.com/app/dashboard/"


def run_miro_login(storage_path: Path = STORAGE_STATE) -> Path:
    """Open Miro in Chromium, wait for user login, save the session.

    Returns the storage path so the caller can echo it.
    """
    print(f"Opening {DASHBOARD_URL} for one-time Miro login…")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        page.goto(DASHBOARD_URL)
        print()
        print("=" * 64)
        print("Log in to Miro in the browser window.")
        print("Once you can see your dashboard, come back to this terminal.")
        print()
        input("Press Enter to save the session and exit > ")
        print("=" * 64)
        ctx.storage_state(path=str(storage_path))
        browser.close()

    print(f"Saved session to {storage_path}")
    return storage_path


if __name__ == "__main__":
    load_dotenv()
    run_miro_login()
    print("Future runs will reuse this. Delete the file to re-authenticate.")
