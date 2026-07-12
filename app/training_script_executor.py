"""
training_script_executor.py

Executes a generated training script as a real subprocess (not a notebook
kernel -- there's no notebook here, just one .py file) and captures
success/failure with enough detail to drive repair: stdout, stderr,
return code, and a best-effort extracted traceback. Also applies the
quality gate: even a script that runs successfully is only a real
"success" if the trained model meaningfully beats a naive baseline.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Model must beat baseline by more than this many of the model's own CV
# standard deviations (a lightweight significance-ish check against noise),
# AND by at least this relative fraction of the baseline's score (so a
# tiny std on a near-perfect baseline doesn't let a trivial improvement pass).
QUALITY_GATE_STD_MULTIPLIER = 1.0
QUALITY_GATE_MIN_RELATIVE_IMPROVEMENT = 0.02


@dataclass
class ExecutionResult:
    """Outcome of running a training script."""

    success: bool
    crashed: bool
    timed_out: bool
    stdout: str
    stderr: str
    metrics: dict | None = None
    quality_gate_passed: bool | None = None
    quality_gate_reason: str | None = None


class TrainingScriptExecutor:
    """Runs a training script as a subprocess and evaluates its output."""

    def __init__(self, timeout_seconds: int = 1800) -> None:
        self._timeout_seconds = timeout_seconds

    def execute(self, script_path: Path, output_dir: Path) -> ExecutionResult:
        """
        Run the script at script_path via a fresh Python subprocess.

        Args:
            script_path: Path to the .py file to run.
            output_dir: Where the script is expected to write metrics.json
                (used to load and quality-gate the result after a clean run).

        Returns:
            An ExecutionResult describing crash/timeout/success and,
            on success, whether the quality gate passed.
        """
        logger.info("Executing training script: %s (timeout=%ds)", script_path, self._timeout_seconds)

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            logger.error("Training script timed out after %ds.", self._timeout_seconds)
            return ExecutionResult(
                success=False, crashed=False, timed_out=True,
                stdout=exc.stdout or "", stderr=exc.stderr or "",
            )

        if proc.returncode != 0:
            logger.error("Training script crashed (exit code %d).", proc.returncode)
            return ExecutionResult(
                success=False, crashed=True, timed_out=False,
                stdout=proc.stdout, stderr=proc.stderr,
            )

        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            logger.error("Script exited 0 but metrics.json was not created at %s.", metrics_path)
            return ExecutionResult(
                success=False, crashed=True, timed_out=False,
                stdout=proc.stdout,
                stderr=proc.stderr + "\n\n[pipeline note: script exited successfully but did not write metrics.json to OUTPUT_DIR as required]",
            )

        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("metrics.json is not valid JSON: %s", exc)
            return ExecutionResult(
                success=False, crashed=True, timed_out=False,
                stdout=proc.stdout,
                stderr=proc.stderr + f"\n\n[pipeline note: metrics.json exists but is not valid JSON: {exc}]",
            )

        gate_passed, gate_reason = _evaluate_quality_gate(metrics)
        logger.info(
            "Script executed successfully. Quality gate: %s (%s)",
            "PASSED" if gate_passed else "FAILED", gate_reason,
        )

        return ExecutionResult(
            success=gate_passed, crashed=False, timed_out=False,
            stdout=proc.stdout, stderr=proc.stderr,
            metrics=metrics, quality_gate_passed=gate_passed, quality_gate_reason=gate_reason,
        )


def _evaluate_quality_gate(metrics: dict) -> tuple[bool, str]:
    """
    Decide whether a successfully-trained model is actually good enough
    to publish, distinct from just "did the script crash or not."

    Passes only if the model beats the baseline in the correct direction
    by more than QUALITY_GATE_STD_MULTIPLIER of the model's own CV std
    (a lightweight guard against the improvement being noise), AND by at
    least QUALITY_GATE_MIN_RELATIVE_IMPROVEMENT relative to the baseline
    (so a tiny std on an already-strong baseline doesn't let a trivial
    improvement pass).
    """
    try:
        direction = metrics["metric_direction"]
        model_mean = float(metrics["model_cv_score_mean"])
        model_std = float(metrics["model_cv_score_std"])
        baseline_mean = float(metrics["baseline_cv_score_mean"])
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"metrics.json missing/invalid required fields: {exc}"

    if direction == "higher_is_better":
        raw_improvement = model_mean - baseline_mean
    elif direction == "lower_is_better":
        raw_improvement = baseline_mean - model_mean
    else:
        return False, f"Unknown metric_direction: {direction!r}"

    std_threshold = QUALITY_GATE_STD_MULTIPLIER * model_std
    relative_threshold = QUALITY_GATE_MIN_RELATIVE_IMPROVEMENT * abs(baseline_mean)
    required = max(std_threshold, relative_threshold)

    if raw_improvement <= 0:
        return False, (
            f"Model ({model_mean:.4f}) did not beat baseline ({baseline_mean:.4f}) "
            f"in the required direction ({direction})."
        )
    if raw_improvement < required:
        return False, (
            f"Model beat baseline by {raw_improvement:.4f}, but this is within noise/too small "
            f"(required > {required:.4f}, i.e. max of {QUALITY_GATE_STD_MULTIPLIER}x model std={std_threshold:.4f} "
            f"or {QUALITY_GATE_MIN_RELATIVE_IMPROVEMENT:.0%} relative improvement={relative_threshold:.4f})."
        )

    return True, (
        f"Model beat baseline by {raw_improvement:.4f} "
        f"(required > {required:.4f}). Model: {model_mean:.4f}, Baseline: {baseline_mean:.4f}."
    )