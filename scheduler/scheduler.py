"""
scheduler/scheduler.py

Runs the full data pipeline every 2 hours, and the LLM morning digest
every day at 8:00 AM.

Run this with:
    python scheduler/scheduler.py

Keep it running in the background (use a terminal tab, tmux, screen,
or Windows Task Scheduler / a service manager for production).
"""

import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import schedule

# ---------------------------------------------------------------------------
# Path setup — lets this script import sibling packages (collectors, etc.)
# regardless of where it's launched from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging — writes to both console and logs/scheduler.log
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scheduler.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("scheduler")

# List of stocks the pipeline tracks. Update this in one place.
TICKERS = ["AAPL", "TSLA", "MSFT", "NVDA", "AMZN"]


# ---------------------------------------------------------------------------
# Job 1: Full pipeline — collectors -> sentiment scoring
# Runs every 2 hours.
# ---------------------------------------------------------------------------
def run_pipeline():
    logger.info("=" * 60)
    logger.info("PIPELINE START")

    # --- Collectors ---
    try:
        from collectors import news_collector
        logger.info("Running news_collector...")
        news_collector.run(TICKERS)
        logger.info("news_collector finished OK")
    except Exception:
        logger.error("news_collector FAILED:\n%s", traceback.format_exc())

    try:
        from collectors import stocktwits_collector
        logger.info("Running stocktwits_collector...")
        stocktwits_collector.run(TICKERS)
        logger.info("stocktwits_collector finished OK")
    except Exception:
        logger.error("stocktwits_collector FAILED:\n%s", traceback.format_exc())

    try:
        from collectors import stock_collector
        logger.info("Running stock_collector...")
        stock_collector.run(TICKERS)
        logger.info("stock_collector finished OK")
    except Exception:
        logger.error("stock_collector FAILED:\n%s", traceback.format_exc())

    # --- Sentiment scoring (FinBERT) on whatever new rows came in ---
    try:
        from sentiment import finbert_analyzer
        logger.info("Running finbert_analyzer...")
        finbert_analyzer.run()
        logger.info("finbert_analyzer finished OK")
    except Exception:
        logger.error("finbert_analyzer FAILED:\n%s", traceback.format_exc())

    logger.info("PIPELINE END")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Job 2: LLM morning digest
# Runs once a day at 8:00 AM.
# ---------------------------------------------------------------------------
def run_digest():
    logger.info("-" * 60)
    logger.info("DIGEST START")
    try:
        from reports import digest_generator
        digest_generator.generate(TICKERS)
        logger.info("Digest generated OK")
    except Exception:
        logger.error("Digest generation FAILED:\n%s", traceback.format_exc())
    logger.info("DIGEST END")
    logger.info("-" * 60)


# ---------------------------------------------------------------------------
# Wrapper so one job crashing never kills the whole scheduler loop
# ---------------------------------------------------------------------------
def safe_run(job_func, name):
    def wrapped():
        try:
            job_func()
        except Exception:
            logger.error("Unhandled error in job '%s':\n%s", name, traceback.format_exc())
    return wrapped


def main():
    logger.info("Scheduler starting up at %s", datetime.now().isoformat(timespec="seconds"))
    logger.info("Tracking tickers: %s", ", ".join(TICKERS))

    # Run once immediately on startup so you don't wait 2 hours to see data
    logger.info("Running pipeline once immediately on startup...")
    safe_run(run_pipeline, "pipeline")()

    # Schedule recurring jobs
    schedule.every(2).hours.do(safe_run(run_pipeline, "pipeline"))
    schedule.every().day.at("08:00").do(safe_run(run_digest, "digest"))

    logger.info("Scheduled: pipeline every 2 hours, digest daily at 08:00")
    logger.info("Scheduler is now running. Press Ctrl+C to stop.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)  # check every 30s — cheap, no need to check every second
    except KeyboardInterrupt:
        logger.info("Scheduler stopped manually (Ctrl+C). Goodbye.")


if __name__ == "__main__":
    main()