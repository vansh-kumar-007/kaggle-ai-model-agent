"""
training_script_repair.py

Repairs a failed training script by sending Nemotron the full script,
the exact failure (crash traceback OR quality-gate rejection reason),
and the original task context -- and getting back a complete rewrite.
Full-script regeneration, not a targeted patch: the script is short
enough (one file, ~150-300 lines) that regeneration is simpler and more
reliable than a diff/patch mechanism, unlike the notebook pipeline's
many-small-cells design where preserving working cells mattered more.
"""

from __future__ import annotations

import logging
from pathlib import Path

from kaggle_ai_core.dataset_analyzer import DatasetProfile
from kaggle_ai_core.nemotron_client import NemotronClient
from kaggle_ai_core.profile_formatting import render_profile_summary

from app.model_task_planner import ModelPlan
from app.training_script_executor import ExecutionResult
from app.training_script_generator import (
    METRIC_DIRECTION, TrainingScript, _inject_environment_paths, _parse_response, _validate_syntax,
)

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "training_repair_prompt.txt"
MAX_PARSE_RETRIES = 2
CV_FOLDS = 5


def _build_failure_reason(result: ExecutionResult) -> str:
    if result.timed_out:
        return (
            "The script exceeded its execution time budget and did not finish. This is a performance "
            "problem: reduce data volume where reasonable (e.g. sample for very expensive steps), "
            "reduce n_estimators/complexity, or use a more efficient encoding/training approach."
        )
    if result.crashed:
        return f"The script crashed. stderr:\n{result.stderr}\n\nstdout (last portion):\n{result.stdout[-2000:]}"
    if result.quality_gate_passed is False:
        return (
            f"The script ran successfully, but the trained model FAILED the quality gate: "
            f"{result.quality_gate_reason}\n\nFull stdout:\n{result.stdout[-2000:]}"
        )
    return "Unknown failure."


class TrainingScriptRepairer:
    """Regenerates a complete, fixed training script via Nemotron given a failure."""

    def __init__(self, nemotron_client: NemotronClient) -> None:
        self._client = nemotron_client
        self._prompt_template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    def repair(
        self,
        plan: ModelPlan,
        profile: DatasetProfile,
        previous_script: str,
        result: ExecutionResult,
        dataset_dir: Path,
        output_dir: Path,
    ) -> TrainingScript:
        """
        Generate a complete replacement script that fixes the given failure.

        Raises:
            ValueError: if a valid script could not be produced even
                after repair-response parse retries.
        """
        metric_direction = METRIC_DIRECTION[plan.eval_metric]
        failure_reason = _build_failure_reason(result)

        prompt = self._prompt_template
        prompt = prompt.replace("<<DATASET_TITLE>>", plan.dataset_title)
        prompt = prompt.replace("<<DATASET_REF>>", plan.dataset_ref)
        prompt = prompt.replace("<<PROBLEM_TYPE>>", plan.problem_type)
        prompt = prompt.replace("<<TARGET_COLUMN>>", plan.target_column)
        prompt = prompt.replace("<<ALGORITHM>>", plan.algorithm)
        prompt = prompt.replace("<<EVAL_METRIC>>", plan.eval_metric)
        prompt = prompt.replace("<<METRIC_DIRECTION>>", metric_direction)
        prompt = prompt.replace("<<PROFILE_SUMMARY>>", render_profile_summary(profile))
        prompt = prompt.replace("<<FAILURE_REASON>>", failure_reason)
        prompt = prompt.replace("<<PREVIOUS_SCRIPT>>", previous_script)
        prompt = prompt.replace("<<CV_FOLDS>>", str(CV_FOLDS))
        prompt = prompt.replace(
            "<<CV_SPLITTER>>",
            f"sklearn.model_selection.KFold(n_splits={CV_FOLDS}, shuffle=True, random_state=42)"
            if plan.problem_type == "regression"
            else f"sklearn.model_selection.StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True, random_state=42)",
        )

        logger.info("Requesting training script repair (reason: %s)", failure_reason[:150])
        raw_response = self._client.generate(
            prompt=prompt,
            system_instruction=(
                "You are a Kaggle Grandmaster debugging and improving ML training code. "
                "Follow the exact plain-text delimiter format with zero preamble."
            ),
            temperature=0.3,
            max_tokens=6144,
        )

        script, error_reason = self._try_parse(raw_response)
        for attempt in range(MAX_PARSE_RETRIES):
            if script is not None:
                break
            logger.warning("Repair response invalid (%s), retrying (%d/%d).", error_reason, attempt + 1, MAX_PARSE_RETRIES)
            retry_prompt = (
                f"Your previous repair response had a problem: {error_reason}\n\n"
                f"Here was your response:\n\n{raw_response}\n\n"
                "Return the corrected response using the EXACT delimiter format, with valid Python."
            )
            raw_response = self._client.generate(
                prompt=retry_prompt,
                system_instruction="You are fixing a malformed repair response. Follow the exact delimiter format.",
                temperature=0.1,
                max_tokens=6144,
            )
            script, error_reason = self._try_parse(raw_response)

        if script is None:
            raise ValueError(f"Failed to generate a valid repaired script: {error_reason}")

        script.code = _inject_environment_paths(script.code, dataset_dir, output_dir)
        return script

    def _try_parse(self, raw_response: str) -> tuple[TrainingScript | None, str | None]:
        try:
            parsed = _parse_response(raw_response)
        except ValueError as exc:
            return None, str(exc)

        syntax_error = _validate_syntax(parsed["code"])
        if syntax_error is not None:
            return None, syntax_error

        return TrainingScript(code=parsed["code"], approach_notes=parsed["approach_notes"]), None