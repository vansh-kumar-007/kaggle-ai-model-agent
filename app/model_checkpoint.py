"""
model_checkpoint.py

Persists state after the expensive stage (dataset selection, profiling,
planning, script generation) completes, so the pipeline can resume from
the execute/repair/publish stage without redoing that work if interrupted.

Deliberately local to this repo, not kaggle-ai-core: the notebook
pipeline's checkpoint shape (notebook_path, prior_variables) is genuinely
different from what this pipeline needs (script_path, algorithm,
rendered profile text) -- only truly generic modules belong in the
shared package.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelPipelineCheckpoint:
    """Everything needed to resume the execute/repair/publish stage."""

    script_path: str
    dataset_dir: str
    output_dir: str
    dataset_ref: str
    dataset_title: str
    profile_summary: str
    plan_json: str  # ModelPlan.model_dump_json()


def save_checkpoint(checkpoint: ModelPipelineCheckpoint, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(checkpoint), indent=2), encoding="utf-8")
    logger.info("Checkpoint saved to %s", path)


def load_checkpoint(path: Path) -> ModelPipelineCheckpoint | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ModelPipelineCheckpoint(**data)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to load checkpoint from %s: %s", path, exc)
        return None


def clear_checkpoint(path: Path) -> None:
    if path.exists():
        path.unlink()
        logger.info("Checkpoint cleared: %s", path)