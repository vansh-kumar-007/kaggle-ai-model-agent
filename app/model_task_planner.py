"""
model_task_planner.py

Decides WHETHER and HOW to train a model on a profiled dataset, before
any training code is written. Mirrors the notebook pipeline's
analysis_planner.py (structured JSON plan via Nemotron), but scoped to
"is this trainable, and if so what's the plan" rather than "what
notebook sections fit."

Deliberately conservative: if no genuinely usable target column exists,
the plan reports trainable=False rather than forcing a target onto data
that doesn't have one -- callers are expected to fall through to the
next shortlist candidate in that case (see main.py).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from kaggle_ai_core.dataset_analyzer import DatasetProfile
from kaggle_ai_core.nemotron_client import NemotronClient
from kaggle_ai_core.profile_formatting import render_profile_summary
from kaggle_ai_core.utils.helpers import extract_json_from_text

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "model_planner_prompt.txt"
MAX_JSON_REPAIR_ATTEMPTS = 1

CLASSIFICATION_ALGORITHMS = {
    "logistic_regression", "random_forest_classifier", "gradient_boosting_classifier",
    "xgboost_classifier", "lightgbm_classifier", "extra_trees_classifier",
}
REGRESSION_ALGORITHMS = {
    "linear_regression", "ridge_regression", "random_forest_regressor",
    "gradient_boosting_regressor", "xgboost_regressor", "lightgbm_regressor", "extra_trees_regressor",
}
ALL_ALGORITHMS = CLASSIFICATION_ALGORITHMS | REGRESSION_ALGORITHMS
VALID_EVAL_METRICS = {"accuracy", "f1_macro", "roc_auc", "rmse", "r2"}
VALID_PROBLEM_TYPES = {"binary_classification", "multiclass_classification", "regression"}


class ModelPlan(BaseModel):
    """Structured plan for whether/how to train a model on one dataset."""

    model_config = ConfigDict(protected_namespaces=())  # allow model_title/model_card_summary field names

    dataset_ref: str
    dataset_title: str
    trainable: bool
    reason_not_trainable: str | None = None

    primary_csv: str | None = None
    problem_type: str | None = None
    target_column: str | None = None
    algorithm: str | None = None
    eval_metric: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    numerical_columns: list[str] = Field(default_factory=list)
    time_columns: list[str] = Field(
        default_factory=list,
        description="Feature columns that are dates/timestamps and need parsing + derived features (year, month, cyclical encoding), not raw categorical/numerical use.",
    )
    columns_to_drop: list[str] = Field(default_factory=list)
    preprocessing_notes: str = ""
    model_title: str = ""
    model_card_summary: str = ""

    @field_validator("problem_type", "algorithm", "eval_metric", mode="before")
    @classmethod
    def _normalize_enum_case(cls, value):
        """
        Normalize case/whitespace on enum-like fields before validating
        membership. An LLM writing 'RMSE' instead of 'rmse' means exactly
        the same thing semantically -- rejecting it and burning a repair
        round-trip over a trivial case difference wastes a Nemotron call
        for no real benefit.
        """
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @model_validator(mode="after")
    def _validate_trainable_fields(self) -> "ModelPlan":
        if not self.trainable:
            return self

        if self.problem_type not in VALID_PROBLEM_TYPES:
            raise ValueError(f"Invalid problem_type: {self.problem_type!r}")
        if self.eval_metric not in VALID_EVAL_METRICS:
            raise ValueError(f"Invalid eval_metric: {self.eval_metric!r}")
        if self.algorithm not in ALL_ALGORITHMS:
            raise ValueError(f"Invalid algorithm: {self.algorithm!r}")

        is_classification = self.problem_type in ("binary_classification", "multiclass_classification")
        if is_classification and self.algorithm not in CLASSIFICATION_ALGORITHMS:
            raise ValueError(f"algorithm '{self.algorithm}' is not valid for problem_type '{self.problem_type}'")
        if self.problem_type == "regression" and self.algorithm not in REGRESSION_ALGORITHMS:
            raise ValueError(f"algorithm '{self.algorithm}' is not valid for problem_type 'regression'")

        if not self.target_column:
            raise ValueError("target_column is required when trainable is true.")
        if not self.primary_csv:
            raise ValueError("primary_csv is required when trainable is true.")

        classified = set(self.categorical_columns) | set(self.numerical_columns) | set(self.time_columns)
        unclassified = set(self.feature_columns) - classified
        if unclassified:
            raise ValueError(
                f"feature_columns contains columns not classified in categorical_columns, "
                f"numerical_columns, or time_columns: {sorted(unclassified)}"
            )

        return self


class ModelTaskPlanner:
    """Generates and validates a ModelPlan via Nemotron for a single profiled dataset."""

    def __init__(self, nemotron_client: NemotronClient) -> None:
        self._client = nemotron_client
        self._prompt_template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    def generate_plan(self, profile: DatasetProfile) -> ModelPlan:
        """
        Generate and validate a ModelPlan for the given dataset profile.

        Raises:
            ValueError: if a valid plan could not be produced even after
                one repair attempt.
        """
        prompt = self._prompt_template
        prompt = prompt.replace("<<DATASET_TITLE>>", profile.title)
        prompt = prompt.replace("<<DATASET_REF>>", profile.ref)
        prompt = prompt.replace("<<PROFILE_SUMMARY>>", render_profile_summary(profile))

        logger.info("Requesting model plan from Nemotron for '%s'", profile.ref)
        raw_response = self._client.generate(
            prompt=prompt,
            system_instruction=(
                "You are a Kaggle Grandmaster deciding on a modeling approach. "
                "Respond with ONLY valid JSON, no markdown fences, no preamble."
            ),
            temperature=0.3,
            max_tokens=2048,
        )

        plan = self._try_parse_plan(raw_response, profile)
        if plan is not None:
            return plan

        logger.warning("Initial model plan response invalid, attempting repair.")
        for attempt in range(MAX_JSON_REPAIR_ATTEMPTS):
            repair_prompt = (
                "Your previous response was not valid JSON matching the required schema, "
                "or used an algorithm/problem_type/eval_metric value outside the allowed lists. "
                f"Here was your response:\n\n{raw_response}\n\n"
                "Return ONLY corrected valid JSON matching the schema exactly, nothing else."
            )
            raw_response = self._client.generate(
                prompt=repair_prompt,
                system_instruction="You are fixing malformed JSON. Respond with ONLY valid JSON.",
                temperature=0.1,
                max_tokens=2048,
            )
            plan = self._try_parse_plan(raw_response, profile)
            if plan is not None:
                logger.info("Model plan repaired successfully on attempt %d.", attempt + 1)
                return plan

        raise ValueError(f"Failed to generate a valid model plan for '{profile.ref}' after retries.")

    def _try_parse_plan(self, raw_response: str, profile: DatasetProfile) -> ModelPlan | None:
        try:
            json_str = extract_json_from_text(raw_response)
            data = json.loads(json_str)
            plan = ModelPlan.model_validate(data)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Model plan parsing/validation failed: %s", exc)
            return None

        if plan.trainable:
            logger.info(
                "Model plan validated: trainable=True, problem_type=%s, target=%s, algorithm=%s",
                plan.problem_type, plan.target_column, plan.algorithm,
            )
        else:
            logger.info("Model plan validated: trainable=False (%s)", plan.reason_not_trainable)
        return plan