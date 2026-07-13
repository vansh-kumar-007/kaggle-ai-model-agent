"""
resume_pipeline.py

Resumes an interrupted model-pipeline run from its last checkpoint --
skips dataset selection, planning, and script generation entirely, and
continues directly from the execute -> repair -> publish loop.
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from kaggle_ai_core.config import LOGS_DIR, PUBLISHED_REFS_PATH, get_settings
from kaggle_ai_core.github_logger import write_run_log
from kaggle_ai_core.kaggle_client import KaggleClient
from kaggle_ai_core.nemotron_client import NemotronClient

from app.main import CHECKPOINT_PATH, run_execute_repair_publish
from app.model_checkpoint import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("resume_pipeline")


def main() -> None:
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    if checkpoint is None:
        print(f"No checkpoint found at {CHECKPOINT_PATH}. Nothing to resume.")
        sys.exit(1)

    logger.info("Resuming from checkpoint: dataset=%s, script=%s", checkpoint.dataset_ref, checkpoint.script_path)

    settings = get_settings()
    kaggle_client = KaggleClient()
    nemotron_client = NemotronClient(settings)

    summary = run_execute_repair_publish(kaggle_client, nemotron_client, settings, checkpoint, PUBLISHED_REFS_PATH)
    write_run_log(summary, LOGS_DIR)

    if summary.success:
        logger.info("Resumed pipeline completed successfully: %s", summary.kernel_url)
        sys.exit(0)
    else:
        logger.error("Resumed pipeline failed again: %s", summary.failure_reason)
        sys.exit(1)


if __name__ == "__main__":
    main()