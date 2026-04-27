"""Per-run logging: tee stdout+stderr into papers/runs/<ts>_<slug>.log.

main.py wraps each topic's pipeline in `with run_log(topic) as path:`. All
print() output (which is the entire reasoning thread — review turns, CU
sub-loop reasoning, miro plan dispatch) lands in that file alongside the live
terminal output, so a demo run leaves a complete artifact behind.

No codebase rewrite needed: existing print() calls pass through the tee
naturally. Code that uses logging.getLogger() also flows through because
StreamHandler resolves sys.stderr at emit time, not at handler construction.
"""

from __future__ import annotations

import re
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent / "papers" / "runs"


class _Tee:
    """Forward writes to a primary stream and a secondary file.

    Failures on the secondary are swallowed — a logging glitch must never
    break the pipeline. Other stream-protocol methods proxy to the primary
    so libraries that probe isatty() / fileno() still get a real terminal.
    """

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, data):
        n = self._primary.write(data)
        try:
            self._secondary.write(data)
            self._secondary.flush()
        except Exception:
            pass
        return n

    def flush(self):
        self._primary.flush()
        try:
            self._secondary.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self._primary, "isatty", lambda: False)()

    def fileno(self):
        return self._primary.fileno()


def slug(topic: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", topic.strip().lower()).strip("-")
    return s[:60] or "run"


@contextmanager
def run_log(topic: str):
    """Tee stdout+stderr to papers/runs/<ts>_<slug>.log for the duration.

    Yields the log file path so the caller can echo it to the user.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = LOG_DIR / f"{ts}_{slug(topic)}.log"

    f = path.open("w", encoding="utf-8", buffering=1)
    f.write(
        f"# curator run log\n"
        f"# topic: {topic}\n"
        f"# started: {datetime.now().isoformat()}\n\n"
    )

    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(real_out, f)
    sys.stderr = _Tee(real_err, f)
    try:
        yield path
    except BaseException:
        # Crash-time traceback: write into the file directly so it survives
        # after we restore the real streams. Python's default handler will
        # still print to the terminal once after the with-block exits — so
        # the user sees a single traceback there, and the file has a copy.
        traceback.print_exc(file=f)
        raise
    finally:
        sys.stdout, sys.stderr = real_out, real_err
        f.write(f"\n# ended: {datetime.now().isoformat()}\n")
        f.close()
