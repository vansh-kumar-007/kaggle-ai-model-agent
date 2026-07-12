"""
kaggle_model_publisher.py

Publishes a trained, quality-gated model to Kaggle Models: creates the
top-level Model entity, then a Model Instance (uploading model.joblib +
metrics.json + a generated model card), using the schema confirmed
against the real Kaggle API in Step 1's discovery test.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from kaggle_ai_core.kaggle_client import KaggleClient
from kaggle_ai_core.utils.file_utils import add_published_ref
from kaggle_ai_core.utils.helpers import slugify

from app.model_task_planner import ModelPlan

logger = logging.getLogger(__name__)

# Maps our internal algorithm names to Kaggle's Model Instance "framework"
# field. Only "scikitLearn" has been confirmed against the real API so far
# (Step 1 discovery test); xgboost/lightGBM values are Kaggle's documented
# UI options but UNVERIFIED against this specific API version -- if wrong,
# Kaggle's own validation error will surface clearly on first real use.
# Confirmed against Kaggle's real API error message (the full valid enum is:
# tensorFlow1, tensorFlow2, tfLite, tfJs, pyTorch, jax, flax, pax, maxText,
# gemmaCpp, tensorRtLlm, ggml, gguf, coral, scikitLearn, mxnet, onnx, keras,
# transformers, triton, other). Kaggle Models has no dedicated slot for
# XGBoost or LightGBM specifically -- "other" is the correct, honest choice
# for those; the actual library used is still documented in the model card
# text itself (see _build_model_card), so nothing is lost, just not captured
# in this one structured field.
FRAMEWORK_MAP = {
    "logistic_regression": "scikitLearn",
    "random_forest_classifier": "scikitLearn",
    "gradient_boosting_classifier": "scikitLearn",
    "extra_trees_classifier": "scikitLearn",
    "linear_regression": "scikitLearn",
    "ridge_regression": "scikitLearn",
    "random_forest_regressor": "scikitLearn",
    "gradient_boosting_regressor": "scikitLearn",
    "extra_trees_regressor": "scikitLearn",
    "xgboost_classifier": "other",
    "xgboost_regressor": "other",
    "lightgbm_classifier": "other",
    "lightgbm_regressor": "other",
}


def _build_model_card(plan: ModelPlan, metrics: dict) -> str:
    """Build the Model's description markdown from REAL computed metrics, not just LLM claims."""
    return (
        f"# Model Summary\n\n{plan.model_card_summary}\n\n"
        f"# Model Characteristics\n\n"
        f"- **Algorithm:** {metrics['algorithm']}\n"
        f"- **Problem type:** {metrics['problem_type']}\n"
        f"- **Target column:** {plan.target_column}\n"
        f"- **Training samples:** {metrics['n_samples']:,}\n"
        f"- **Features:** {metrics['n_features']}\n\n"
        f"# Evaluation Results\n\n"
        f"Evaluated via {metrics['cv_folds']}-fold cross-validation, compared against a naive baseline.\n\n"
        f"| | Model | Baseline |\n"
        f"|---|---|---|\n"
        f"| {metrics['eval_metric'].upper()} (mean) | {metrics['model_cv_score_mean']:.4f} | {metrics['baseline_cv_score_mean']:.4f} |\n"
        f"| {metrics['eval_metric'].upper()} (std) | {metrics['model_cv_score_std']:.4f} | {metrics['baseline_cv_score_std']:.4f} |\n\n"
        f"Metric direction: {metrics['metric_direction'].replace('_', ' ')}.\n"
    )


def _build_usage_section(plan: ModelPlan) -> str:
    return (
        f"# Model Format\n\nSerialized via joblib (scikit-learn-compatible estimator/pipeline).\n\n"
        f"# Training Data\n\nTrained on the Kaggle dataset: {plan.dataset_ref}\n\n"
        f"# Model Inputs\n\nFeature columns: {', '.join(plan.feature_columns)}\n\n"
        f"# Model Outputs\n\nPredicts: {plan.target_column} ({plan.problem_type})\n\n"
        f"# Model Usage\n\n```python\nimport joblib\nmodel = joblib.load('model.joblib')\npredictions = model.predict(X)\n```\n\n"
        f"# Fine-tuning\n\nNot applicable -- this is a fitted classical ML estimator, not a fine-tunable base model.\n\n"
        f"# Changelog\n\nInitial version, generated and trained automatically.\n"
    )


class KaggleModelPublisher:
    """Publishes a trained model + metrics to Kaggle Models."""

    def __init__(self, kaggle_client: KaggleClient, kaggle_username: str, publish_workdir: Path) -> None:
        self._client = kaggle_client
        self._username = kaggle_username
        self._publish_workdir = publish_workdir

    def publish(
        self,
        plan: ModelPlan,
        model_output_dir: Path,
        published_refs_path: Path,
        is_private: bool = False,
    ) -> str:
        """
        Publish a trained model to Kaggle Models.

        Args:
            plan: The ModelPlan that produced this model (for title/card content).
            model_output_dir: Directory containing model.joblib and metrics.json.
            published_refs_path: JSON file tracking published dataset refs
                (shared duplicate-avoidance mechanism with the notebook pipeline's
                DatasetSelector, using the SAME file if pointed at the same path --
                or a separate model-specific file, depending on how this is wired
                in main.py).
            is_private: Whether the Kaggle Model should be private. Defaults False.

        Returns:
            The public URL of the published Kaggle Model.

        Raises:
            FileNotFoundError: if model.joblib or metrics.json are missing.
            Exception: propagated from the Kaggle API on failure (e.g. an
                unverified framework mapping being rejected).
        """
        model_path = model_output_dir / "model.joblib"
        metrics_path = model_output_dir / "metrics.json"
        if not model_path.exists():
            raise FileNotFoundError(f"model.joblib not found at {model_path}")
        if not metrics_path.exists():
            raise FileNotFoundError(f"metrics.json not found at {metrics_path}")

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        framework = FRAMEWORK_MAP.get(metrics["algorithm"])
        if framework is None:
            raise ValueError(f"No framework mapping for algorithm '{metrics['algorithm']}'.")

        model_slug = slugify(plan.model_title, max_length=50)

        # --- Step A: create the top-level Model entity ---
        model_dir = self._publish_workdir / model_slug / "model"
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True)

        model_metadata = {
            "ownerSlug": self._username,
            "title": plan.model_title[:50],
            "slug": model_slug,
            "subtitle": plan.model_card_summary[:200],
            "isPrivate": is_private,
            "description": _build_model_card(plan, metrics),
            "publishTime": "",
            "provenanceSources": f"Trained on Kaggle dataset: {plan.dataset_ref}",
        }
        (model_dir / "model-metadata.json").write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")

        logger.info("Creating Kaggle Model entity: %s/%s", self._username, model_slug)
        self._client.create_model(model_dir)

        # --- Step B: create the Model Instance with the actual files ---
        instance_dir = self._publish_workdir / model_slug / "instance"
        if instance_dir.exists():
            shutil.rmtree(instance_dir)
        instance_dir.mkdir(parents=True)

        shutil.copy2(model_path, instance_dir / "model.joblib")
        shutil.copy2(metrics_path, instance_dir / "metrics.json")

        instance_metadata = {
            "ownerSlug": self._username,
            "modelSlug": model_slug,
            "instanceSlug": "default",
            "framework": framework,
            "overview": plan.model_card_summary,
            "usage": _build_usage_section(plan),
            "licenseName": "Apache 2.0",
            "fineTunable": False,
            "trainingData": [plan.dataset_ref],
            "modelInstanceType": "Unspecified",
            "baseModelInstanceId": 0,
            "externalBaseModelUrl": "",
        }
        (instance_dir / "model-instance-metadata.json").write_text(
            json.dumps(instance_metadata, indent=2), encoding="utf-8"
        )

        logger.info("Creating Kaggle Model Instance (uploading model.joblib + metrics.json)...")
        self._client.create_model_instance(instance_dir)

        add_published_ref(published_refs_path, plan.dataset_ref)

        model_url = f"https://www.kaggle.com/models/{self._username}/{model_slug}"
        logger.info("Published successfully: %s", model_url)
        return model_url