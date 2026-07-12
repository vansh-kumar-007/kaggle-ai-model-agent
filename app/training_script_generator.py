"""
training_script_generator.py

Generates ONE complete, self-contained Python training script from a
validated ModelPlan -- load data, preprocess per the plan's column
buckets, cross-validate the chosen algorithm against a naive baseline,
fit a final model on all data, and serialize model + metrics to
OUTPUT_DIR. Uses the same plain-text delimiter protocol (not JSON) that
the notebook pipeline's section_generators.py adopted, for the same
reason: raw Python code (quotes, backslashes, docstrings) breaks JSON
string escaping unpredictably.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from kaggle_ai_core.dataset_analyzer import DatasetProfile
from kaggle_ai_core.nemotron_client import NemotronClient
from kaggle_ai_core.profile_formatting import render_profile_summary

from app.model_task_planner import ModelPlan

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "training_script_prompt.txt"
MAX_REPAIR_ATTEMPTS = 2

CV_FOLDS = 5

# Fixed, deterministic mapping -- NOT decided by the LLM, to avoid ever
# getting a quality-gate direction backwards.
METRIC_DIRECTION = {
    "accuracy": "higher_is_better",
    "f1_macro": "higher_is_better",
    "roc_auc": "higher_is_better",
    "r2": "higher_is_better",
    "rmse": "lower_is_better",
}


_ENV_VAR_ASSIGNMENT_PATTERN = re.compile(r"^(DATASET_DIR|OUTPUT_DIR)\s*=.*$", re.MULTILINE)


def _inject_environment_paths(code: str, dataset_dir: Path, output_dir: Path) -> str:
    """
    Guarantee DATASET_DIR and OUTPUT_DIR resolve to the real, correct
    local paths, regardless of what (if anything) the LLM assigned to
    them. Mirrors the notebook pipeline's deterministic setup-cell
    approach: infra values that MUST be correct are never trusted to the
    LLM. Any DATASET_DIR/OUTPUT_DIR assignment line the LLM wrote is
    neutralized (commented out) so it can't override our injected value
    later in the script -- Python executes top-to-bottom, so a later
    reassignment would silently win otherwise.
    """
    dataset_dir_str = str(dataset_dir).replace("\\", "/")
    output_dir_str = str(output_dir).replace("\\", "/")

    neutralized = _ENV_VAR_ASSIGNMENT_PATTERN.sub(
        r"# \g<1> assignment removed -- injected deterministically below instead",
        code,
    )

    preamble = (
        "# --- Injected by pipeline: guaranteed-correct paths, do not trust LLM-written values above/below ---\n"
        "from pathlib import Path as _PipelinePath\n"
        f'DATASET_DIR = _PipelinePath(r"{dataset_dir_str}")\n'
        f'OUTPUT_DIR = _PipelinePath(r"{output_dir_str}")\n'
        "# --- End injected preamble ---\n\n"
    )
    return preamble + neutralized


_SCRIPT_PATTERN = re.compile(r"===SCRIPT_CODE===\s*\n(.*?)\n?===END_SCRIPT===", re.DOTALL)
_NOTES_PATTERN = re.compile(r"===APPROACH_NOTES===\s*\n(.*?)(?:\n===END===|\Z)", re.DOTALL)


@dataclass
class TrainingScript:
    """A generated, syntax-validated training script."""

    code: str
    approach_notes: str


def _cv_splitter_description(problem_type: str) -> str:
    if problem_type == "regression":
        return f"sklearn.model_selection.KFold(n_splits={CV_FOLDS}, shuffle=True, random_state=42)"
    return f"sklearn.model_selection.StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True, random_state=42)"


def _parse_response(raw_response: str) -> dict:
    script_match = _SCRIPT_PATTERN.search(raw_response)
    if not script_match:
        raise ValueError("Missing ===SCRIPT_CODE=== block in response.")
    code = script_match.group(1).strip()
    if not code:
        raise ValueError("===SCRIPT_CODE=== block is empty.")

    notes_match = _NOTES_PATTERN.search(raw_response)
    notes = notes_match.group(1).strip() if notes_match else ""

    return {"code": code, "approach_notes": notes}


def _validate_syntax(code: str) -> str | None:
    try:
        compile(code, "<training_script>", "exec")
    except SyntaxError as exc:
        return f"Syntax error: {exc.msg} (line {exc.lineno}: {exc.text!r})"
    return None


class TrainingScriptGenerator:
    """Generates and validates a complete training script via Nemotron."""

    def __init__(self, nemotron_client: NemotronClient) -> None:
        self._client = nemotron_client
        self._prompt_template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    def generate(self, plan: ModelPlan, profile: DatasetProfile, dataset_dir: Path, output_dir: Path) -> TrainingScript:
        """
        Generate a complete training script for a trainable ModelPlan.

        Args:
            dataset_dir: Real local directory containing the dataset's CSVs.
            output_dir: Real local directory to save model.joblib/metrics.json into.

        Raises:
            ValueError: if plan.trainable is False, or a valid script
                could not be produced even after repair attempts.
        """
        if not plan.trainable:
            raise ValueError("Cannot generate a training script for a non-trainable plan.")

        metric_direction = METRIC_DIRECTION[plan.eval_metric]

        prompt = self._prompt_template
        prompt = prompt.replace("<<DATASET_TITLE>>", plan.dataset_title)
        prompt = prompt.replace("<<DATASET_REF>>", plan.dataset_ref)
        prompt = prompt.replace("<<PROBLEM_TYPE>>", plan.problem_type)
        prompt = prompt.replace("<<TARGET_COLUMN>>", plan.target_column)
        prompt = prompt.replace("<<ALGORITHM>>", plan.algorithm)
        prompt = prompt.replace("<<EVAL_METRIC>>", plan.eval_metric)
        prompt = prompt.replace("<<METRIC_DIRECTION>>", metric_direction)
        prompt = prompt.replace("<<PRIMARY_CSV>>", plan.primary_csv)
        prompt = prompt.replace("<<FEATURE_COLUMNS>>", ", ".join(plan.feature_columns))
        prompt = prompt.replace("<<CATEGORICAL_COLUMNS>>", ", ".join(plan.categorical_columns) or "(none)")
        prompt = prompt.replace("<<NUMERICAL_COLUMNS>>", ", ".join(plan.numerical_columns) or "(none)")
        prompt = prompt.replace("<<TIME_COLUMNS>>", ", ".join(plan.time_columns) or "(none)")
        prompt = prompt.replace("<<COLUMNS_TO_DROP>>", ", ".join(plan.columns_to_drop) or "(none)")
        prompt = prompt.replace("<<PREPROCESSING_NOTES>>", plan.preprocessing_notes)
        prompt = prompt.replace("<<PROFILE_SUMMARY>>", render_profile_summary(profile))
        prompt = prompt.replace("<<CV_FOLDS>>", str(CV_FOLDS))
        prompt = prompt.replace("<<CV_SPLITTER>>", _cv_splitter_description(plan.problem_type))

        logger.info(
            "Requesting training script from Nemotron: algorithm=%s, target=%s",
            plan.algorithm, plan.target_column,
        )
        raw_response = self._client.generate(
            prompt=prompt,
            system_instruction=(
                "You are a Kaggle Grandmaster writing production-quality ML training code. "
                "Follow the exact plain-text delimiter format with zero preamble. "
                "Do not use JSON or markdown code fences."
            ),
            temperature=0.3,
            max_tokens=6144,
        )

        script, error_reason = self._try_parse(raw_response)
        if script is not None:
            script.code = _inject_environment_paths(script.code, dataset_dir, output_dir)
            return script

        for attempt in range(MAX_REPAIR_ATTEMPTS):
            logger.warning("Training script invalid (%s), retrying (%d/%d).", error_reason, attempt + 1, MAX_REPAIR_ATTEMPTS)
            repair_prompt = (
                f"Your previous response had a problem: {error_reason}\n\n"
                f"Here was your response:\n\n{raw_response}\n\n"
                "Return the corrected response using the EXACT delimiter format described earlier, "
                "with valid, syntactically correct, complete Python. No explanation outside ===APPROACH_NOTES==="
            )
            raw_response = self._client.generate(
                prompt=repair_prompt,
                system_instruction="You are fixing a malformed training script response. Follow the exact delimiter format.",
                temperature=0.1,
                max_tokens=6144,
            )
            script, error_reason = self._try_parse(raw_response)
            if script is not None:
                logger.info("Training script repaired successfully on attempt %d.", attempt + 1)
                script.code = _inject_environment_paths(script.code, dataset_dir, output_dir)
                return script

        raise ValueError(f"Failed to generate a valid training script after retries: {error_reason}")

    def _try_parse(self, raw_response: str) -> tuple[TrainingScript | None, str | None]:
        try:
            parsed = _parse_response(raw_response)
        except ValueError as exc:
            return None, str(exc)

        syntax_error = _validate_syntax(parsed["code"])
        if syntax_error is not None:
            return None, syntax_error

        return TrainingScript(code=parsed["code"], approach_notes=parsed["approach_notes"]), None