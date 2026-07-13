"""
main.py

Full pipeline orchestrator for the model-publishing pipeline -- the
entry point GitHub Actions calls daily. Selects a trainable dataset
(with fallback through the shortlist), plans and generates a training
script, checkpoints, executes it with repair-on-failure (bounded by a
wall-clock budget, not an attempt cap -- Nemotron access is currently
free), and publishes the resulting model to Kaggle.

Run locally with: python -m app.main
Resume an interrupted run with: python resume_pipeline.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from kaggle_ai_core.config import DATASETS_DIR, LOGS_DIR, PUBLISHED_REFS_PATH, ensure_directories, get_settings
from kaggle_ai_core.dataset_analyzer import DatasetAnalyzer
from kaggle_ai_core.dataset_selector import DatasetSelector
from kaggle_ai_core.github_logger import RunSummary, write_run_log
from kaggle_ai_core.kaggle_client import KaggleClient
from kaggle_ai_core.nemotron_client import NemotronClient
from kaggle_ai_core.profile_formatting import render_profile_summary

from app.kaggle_model_publisher import KaggleModelPublisher
from app.model_checkpoint import ModelPipelineCheckpoint, clear_checkpoint, save_checkpoint
from app.model_task_planner import ModelPlan, ModelTaskPlanner
from app.training_script_executor import TrainingScriptExecutor
from app.training_script_generator import TrainingScriptGenerator
from app.training_script_repair import TrainingScriptRepairer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("main")

REPAIR_LOOP_BUDGET_SECONDS = 5 * 3600 + 45 * 60  # 5h45m, same reasoning as notebook-agent
PER_EXECUTION_TIMEOUT_SECONDS = 1800
MODELS_OUT_DIR = Path("models_out")
CHECKPOINT_PATH = LOGS_DIR / "model_pipeline_checkpoint.json"


def run_execute_repair_publish(
    kaggle_client: KaggleClient,
    nemotron_client: NemotronClient,
    settings,
    checkpoint: ModelPipelineCheckpoint,
    published_refs_path: Path,
) -> RunSummary:
    """
    Runs the execute -> repair -> publish loop for an already-generated
    training script described by `checkpoint`. Shared by a fresh run
    and resume_pipeline.py.

    Note: reuses kaggle_ai_core's RunSummary (built for the notebook
    pipeline) -- its `notebook_title`/`kernel_url` fields hold this
    pipeline's model title/model URL instead. Field names don't perfectly
    match semantically, but the shape (dataset ref/title, attempts,
    repairs, success, failure reason) is genuinely identical, so this
    reuses the shared shape rather than duplicating a near-copy.
    """
    summary = RunSummary()
    summary.dataset_ref = checkpoint.dataset_ref
    summary.dataset_title = checkpoint.dataset_title

    plan = ModelPlan.model_validate_json(checkpoint.plan_json)
    summary.notebook_title = plan.model_title  # see docstring note above

    script_path = Path(checkpoint.script_path)
    dataset_dir = Path(checkpoint.dataset_dir)
    output_dir = Path(checkpoint.output_dir)

    executor = TrainingScriptExecutor(timeout_seconds=PER_EXECUTION_TIMEOUT_SECONDS)
    repairer = TrainingScriptRepairer(nemotron_client)

    loop_start = time.monotonic()
    attempt = 0

    while True:
        elapsed = time.monotonic() - loop_start
        if elapsed > REPAIR_LOOP_BUDGET_SECONDS:
            summary.execution_attempts = attempt
            summary.mark_finished(
                False,
                f"Repair loop exceeded time budget ({REPAIR_LOOP_BUDGET_SECONDS / 3600:.1f}h) after {attempt} "
                f"attempts. Checkpoint preserved at {CHECKPOINT_PATH} -- run resume_pipeline.py to continue.",
            )
            return summary

        attempt += 1
        logger.info("Execution attempt %d (elapsed: %.0fs)", attempt, elapsed)
        result = executor.execute(script_path, output_dir)

        if result.success:
            break

        reason = "TIMEOUT" if result.timed_out else ("CRASH" if result.crashed else "QUALITY GATE FAILED")
        logger.warning("Attempt %d failed (%s).", attempt, reason)

        current_code = script_path.read_text(encoding="utf-8")
        try:
            repaired = repairer.repair(plan, checkpoint.profile_summary, current_code, result, dataset_dir, output_dir)
        except Exception:
            logger.exception("Repair call raised an unexpected error on attempt %d.", attempt)
            summary.execution_attempts = attempt
            summary.mark_finished(False, f"Repair call failed on attempt {attempt}; checkpoint preserved for resume.")
            return summary

        summary.repairs_applied += 1
        script_path.write_text(repaired.code, encoding="utf-8")

    summary.execution_attempts = attempt
    logger.info("Script executed successfully & passed quality gate after %d attempt(s), %d repair(s).", attempt, summary.repairs_applied)

    publish_workdir = LOGS_DIR / "model_publish_staging"
    publisher = KaggleModelPublisher(kaggle_client, settings.kaggle_username, publish_workdir)
    model_url = publisher.publish(
        plan=plan,
        model_output_dir=output_dir,
        published_refs_path=published_refs_path,
        is_private=False,
    )
    summary.kernel_url = model_url  # see docstring note above
    summary.mark_finished(True)
    clear_checkpoint(CHECKPOINT_PATH)
    return summary


def run_pipeline() -> RunSummary:
    settings = get_settings()
    ensure_directories()
    MODELS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    kaggle_client = KaggleClient()
    nemotron_client = NemotronClient(settings)

    selector = DatasetSelector(kaggle_client, PUBLISHED_REFS_PATH)
    shortlist = selector.get_shortlist(top_n=5)
    if not shortlist:
        summary = RunSummary()
        summary.mark_finished(False, "No dataset passed hard filters today.")
        return summary

    analyzer = DatasetAnalyzer(kaggle_client, DATASETS_DIR)
    profiles = analyzer.analyze_shortlist_all(shortlist)
    if not profiles:
        summary = RunSummary()
        summary.mark_finished(False, "No shortlisted dataset could be successfully profiled.")
        return summary

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
            logger.info("Selected trainable dataset: %s (%s)", profile.title, profile.ref)
            break
        logger.info("'%s' not trainable: %s", profile.ref, plan.reason_not_trainable)

    if chosen_plan is None:
        summary = RunSummary()
        summary.mark_finished(False, "No shortlisted candidate was trainable.")
        return summary

    dataset_dir = DATASETS_DIR / chosen_profile.ref.replace("/", "__")
    output_dir = MODELS_OUT_DIR / chosen_profile.ref.replace("/", "__")
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "train.py"

    profile_summary = render_profile_summary(chosen_profile)

    generator = TrainingScriptGenerator(nemotron_client)
    try:
        script = generator.generate(chosen_plan, profile_summary, dataset_dir, output_dir)
    except ValueError as exc:
        summary = RunSummary()
        summary.dataset_ref = chosen_profile.ref
        summary.dataset_title = chosen_profile.title
        summary.mark_finished(False, f"Training script generation failed: {exc}")
        return summary

    script_path.write_text(script.code, encoding="utf-8")
    logger.info("Training script generated: %s (%d lines)", script_path, len(script.code.splitlines()))

    checkpoint = ModelPipelineCheckpoint(
        script_path=str(script_path),
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        dataset_ref=chosen_profile.ref,
        dataset_title=chosen_profile.title,
        profile_summary=profile_summary,
        plan_json=chosen_plan.model_dump_json(),
    )
    save_checkpoint(checkpoint, CHECKPOINT_PATH)

    return run_execute_repair_publish(kaggle_client, nemotron_client, settings, checkpoint, PUBLISHED_REFS_PATH)


def main() -> None:
    try:
        summary = run_pipeline()
    except Exception as exc:
        logger.exception("Pipeline crashed with an unhandled exception.")
        summary = RunSummary()
        summary.mark_finished(False, f"Unhandled exception: {exc}")

    write_run_log(summary, LOGS_DIR)

    if summary.success:
        logger.info("Pipeline completed successfully: %s", summary.kernel_url)
        sys.exit(0)
    else:
        logger.error("Pipeline failed: %s", summary.failure_reason)
        sys.exit(1)


if __name__ == "__main__":
    main()