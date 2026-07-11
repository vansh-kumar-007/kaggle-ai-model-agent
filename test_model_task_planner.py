"""
test_model_task_planner.py

Manual verification script.

Runs dataset selection, profiles ALL shortlist candidates (not just the
winner), and tries the model task planner on each in order until one
comes back trainable=True -- the fallback chain requested for when the
top-scoring candidate has no usable target column.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

load_dotenv()

from kaggle_ai_core.config import DATASETS_DIR, LOGS_DIR, PUBLISHED_REFS_PATH, ensure_directories, get_settings  # noqa: E402
from kaggle_ai_core.dataset_analyzer import DatasetAnalyzer  # noqa: E402
from kaggle_ai_core.dataset_selector import DatasetSelector  # noqa: E402
from kaggle_ai_core.kaggle_client import KaggleClient  # noqa: E402
from kaggle_ai_core.nemotron_client import NemotronClient  # noqa: E402

from app.model_task_planner import ModelTaskPlanner  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    ensure_directories()
    settings = get_settings()
    kaggle_client = KaggleClient()
    nemotron_client = NemotronClient(settings)

    selector = DatasetSelector(kaggle_client, PUBLISHED_REFS_PATH)
    shortlist = selector.get_shortlist(top_n=5)

    analyzer = DatasetAnalyzer(kaggle_client, DATASETS_DIR)
    profiles = analyzer.analyze_shortlist_all(shortlist)

    print(f"\n{'='*70}")
    print(f"Profiled {len(profiles)} candidate(s), trying each for trainability...")
    print(f"{'='*70}")

    planner = ModelTaskPlanner(nemotron_client)
    chosen_plan = None
    chosen_profile = None

    for profile in profiles:
        print(f"\nTrying: {profile.title} ({profile.ref})")
        plan = planner.generate_plan(profile)
        if plan.trainable:
            chosen_plan = plan
            chosen_profile = profile
            print("  -> TRAINABLE, stopping fallback chain.")
            break
        else:
            print(f"  -> not trainable: {plan.reason_not_trainable}")

    print(f"\n{'='*70}")
    if chosen_plan is None:
        print("No candidate in the shortlist was trainable.")
    else:
        print(f"SELECTED: {chosen_profile.title} ({chosen_profile.ref})")
        print(f"problem_type: {chosen_plan.problem_type}")
        print(f"target_column: {chosen_plan.target_column}")
        print(f"algorithm: {chosen_plan.algorithm}")
        print(f"eval_metric: {chosen_plan.eval_metric}")
        print(f"primary_csv: {chosen_plan.primary_csv}")
        print(f"feature_columns: {chosen_plan.feature_columns}")
        print(f"categorical_columns: {chosen_plan.categorical_columns}")
        print(f"numerical_columns: {chosen_plan.numerical_columns}")
        print(f"time_columns: {chosen_plan.time_columns}")
        print(f"columns_to_drop: {chosen_plan.columns_to_drop}")
        print(f"preprocessing_notes: {chosen_plan.preprocessing_notes}")
        print(f"model_title: {chosen_plan.model_title}")
        print(f"model_card_summary: {chosen_plan.model_card_summary}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()