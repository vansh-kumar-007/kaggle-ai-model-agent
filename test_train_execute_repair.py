"""
test_train_execute_repair.py

Manual verification script for Step 4.

Runs dataset selection -> profiling -> model planning (with fallback) ->
script generation -> execute/repair loop (bounded by wall-clock budget,
no attempt cap) until the script runs clean AND passes the quality gate.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from kaggle_ai_core.config import DATASETS_DIR, PUBLISHED_REFS_PATH, ensure_directories, get_settings  # noqa: E402
from kaggle_ai_core.dataset_analyzer import DatasetAnalyzer  # noqa: E402
from kaggle_ai_core.dataset_selector import DatasetSelector  # noqa: E402
from kaggle_ai_core.kaggle_client import KaggleClient  # noqa: E402
from kaggle_ai_core.nemotron_client import NemotronClient  # noqa: E402

from app.model_task_planner import ModelTaskPlanner  # noqa: E402
from app.training_script_executor import TrainingScriptExecutor  # noqa: E402
from app.training_script_generator import TrainingScriptGenerator  # noqa: E402
from app.training_script_repair import TrainingScriptRepairer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPAIR_BUDGET_SECONDS = 5 * 3600 + 45 * 60
PER_EXECUTION_TIMEOUT_SECONDS = 1800


def main() -> None:
    ensure_directories()
    settings = get_settings()
    kaggle_client = KaggleClient()
    nemotron_client = NemotronClient(settings)

    selector = DatasetSelector(kaggle_client, PUBLISHED_REFS_PATH)
    shortlist = selector.get_shortlist(top_n=5)
    analyzer = DatasetAnalyzer(kaggle_client, DATASETS_DIR)
    profiles = analyzer.analyze_shortlist_all(shortlist)

    planner = ModelTaskPlanner(nemotron_client)
    chosen_plan, chosen_profile = None, None
    for profile in profiles:
        try:
            plan = planner.generate_plan(profile)
        except ValueError as exc:
            logger.warning("Planning failed entirely for '%s', trying next candidate: %s", profile.ref, exc)
            continue
        if plan.trainable:
            chosen_plan, chosen_profile = plan, profile
            break
        else:
            logger.info("'%s' not trainable: %s", profile.ref, plan.reason_not_trainable)

    if chosen_plan is None:
        print("No trainable candidate found.")
        return

    dataset_dir = DATASETS_DIR / chosen_profile.ref.replace("/", "__")
    output_dir = Path("models_out") / chosen_profile.ref.replace("/", "__")
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "train.py"

    generator = TrainingScriptGenerator(nemotron_client)
    script = generator.generate(chosen_plan, chosen_profile, dataset_dir, output_dir)
    script_path.write_text(script.code, encoding="utf-8")

    executor = TrainingScriptExecutor(timeout_seconds=PER_EXECUTION_TIMEOUT_SECONDS)
    repairer = TrainingScriptRepairer(nemotron_client)

    loop_start = time.monotonic()
    attempt = 0
    while True:
        elapsed = time.monotonic() - loop_start
        if elapsed > REPAIR_BUDGET_SECONDS:
            print(f"Repair budget ({REPAIR_BUDGET_SECONDS}s) exceeded after {attempt} attempts.")
            return

        attempt += 1
        print(f"\n{'='*70}\nEXECUTION ATTEMPT {attempt}\n{'='*70}")
        result = executor.execute(script_path, output_dir)

        if result.success:
            print(f"\nSUCCESS after {attempt} attempt(s).")
            print(f"Metrics: {result.metrics}")
            print(f"Quality gate: {result.quality_gate_reason}")
            return

        reason = "TIMEOUT" if result.timed_out else ("CRASH" if result.crashed else "QUALITY GATE FAILED")
        print(f"FAILED ({reason})")
        if result.quality_gate_reason:
            print(f"  {result.quality_gate_reason}")
        elif result.stderr:
            print(f"  {result.stderr[-500:]}")

        repaired = repairer.repair(chosen_plan, chosen_profile, script.code, result, dataset_dir, output_dir)
        script_path.write_text(repaired.code, encoding="utf-8")
        script = repaired
        print(f"Repair applied: {repaired.approach_notes}")


if __name__ == "__main__":
    main()