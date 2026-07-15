"""
test_kaggle_model_publisher.py

Manual verification script for Step 5.

Publishes an already-trained model (from a prior successful
test_train_execute_repair.py run) to Kaggle Models.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from kaggle_ai_core.config import PUBLISHED_REFS_PATH, ensure_directories, get_settings  # noqa: E402
from kaggle_ai_core.kaggle_client import KaggleClient  # noqa: E402
from kaggle_ai_core.nemotron_client import NemotronClient  # noqa: E402
from kaggle_ai_core.dataset_analyzer import DatasetAnalyzer  # noqa: E402
from kaggle_ai_core.config import DATASETS_DIR  # noqa: E402

from app.kaggle_model_publisher import KaggleModelPublisher  # noqa: E402
from app.model_task_planner import ModelPlan  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_output_dir", type=str, help="e.g. models_out/robikscube__hourly-energy-consumption")
    parser.add_argument("dataset_ref", type=str, help="e.g. robikscube/hourly-energy-consumption")
    parser.add_argument("--title", type=str, default="Hourly Energy Demand Forecaster (LightGBM)")
    parser.add_argument("--summary", type=str, default="Predicts hourly energy demand (MW) from temporal features using LightGBM, trained on 121K observations.")
    parser.add_argument("--target", type=str, default="AEP_MW")
    parser.add_argument("--primary-csv", type=str, required=True, help="Real CSV filename the model was trained on, e.g. healthcare_dataset.csv")
    parser.add_argument("--problem-type", type=str, default="regression", dest="problem_type")
    parser.add_argument("--algorithm", type=str, default="lightgbm_regressor")
    parser.add_argument("--eval-metric", type=str, default="rmse", dest="eval_metric")
    args = parser.parse_args()

    ensure_directories()
    settings = get_settings()
    kaggle_client = KaggleClient()

    plan = ModelPlan(
        dataset_ref=args.dataset_ref,
        dataset_title=args.dataset_ref,
        trainable=True,
        primary_csv=args.primary_csv,
        problem_type=args.problem_type,
        target_column=args.target,
        algorithm=args.algorithm,
        eval_metric=args.eval_metric,
        feature_columns=[],
        model_title=args.title,
        model_card_summary=args.summary,
    )

    publisher = KaggleModelPublisher(kaggle_client, settings.kaggle_username, Path("logs") / "model_publish_staging")
    url = publisher.publish(
        plan=plan,
        model_output_dir=Path(args.model_output_dir),
        published_refs_path=PUBLISHED_REFS_PATH,
        is_private=False,
    )

    print(f"\n{'='*70}\nPUBLISHED: {url}\n{'='*70}\n")


if __name__ == "__main__":
    main()