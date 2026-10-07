# retention_scheduler.py
# ─────────────────────────────────────────────────────────────────────
# Runs the retention sweep automatically on a schedule, instead of
# relying on someone remembering to call POST /admin/retention-sweep.
#
# Two ways to use this:
#
# 1. IN-PROCESS (simplest): import and call `start_background_scheduler()`
#    from api.py's lifespan — runs inside the same FastAPI process.
#
# 2. STANDALONE (more robust for production): run this file as its own
#    process via cron, e.g.:
#       0 3 * * *  /usr/bin/python3 /path/to/retention_scheduler.py --once
#    This avoids tying cleanup to API uptime.
# ─────────────────────────────────────────────────────────────────────

import argparse
import logging
import sys
import time
import threading

from config import RETENTION_DAYS, QDRANT_PATH
from embedder import Embedder
from store import StoryStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def run_sweep_standalone() -> int:
    """Opens its own Qdrant connection, runs one sweep, returns count deleted.
    NOTE: Qdrant's local (path=) mode is single-process-access only — do NOT
    run this standalone version while the FastAPI app (which also opens the
    same QDRANT_PATH) is running, or you'll get a lock conflict. If you need
    cleanup running independently of the API process, switch Qdrant to
    server mode (QdrantClient(url=...)) instead of local path mode.
    """
    embedder = Embedder()      # needed only to compute vector dims for collection check
    store = StoryStore(embedder)
    return store.delete_older_than(RETENTION_DAYS)


def start_background_scheduler(matcher, interval_hours: int = 24) -> None:
    """
    IN-PROCESS option: call this once from api.py's lifespan startup.
    Runs the sweep every `interval_hours` in a daemon thread, using the
    SAME matcher/store instance the API already has open — avoids the
    Qdrant local-mode lock conflict described above.
    """
    def _loop():
        while True:
            time.sleep(interval_hours * 3600)
            try:
                deleted = matcher.run_retention_sweep(RETENTION_DAYS)
                logger.info("Scheduled retention sweep: deleted %d stories.", deleted)
            except Exception as e:
                logger.error("Retention sweep failed: %s", e)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    logger.info("Background retention scheduler started (every %d hours).", interval_hours)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run a single sweep and exit (for cron)")
    args = parser.parse_args()

    if args.once:
        deleted = run_sweep_standalone()
        logger.info("Standalone sweep complete: %d stories deleted.", deleted)
    else:
        print("Use --once for cron-triggered runs, or import start_background_scheduler() into api.py.")
