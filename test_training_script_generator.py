"""
test_training_script_generator.py

Manual verification script for Step 3.

Runs dataset selection -> profiling -> model task planning (with
fallback chain) -> training script generation, and prints the resulting
script for manual review. Does NOT execute the script -- that's Step 4.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

load_dotenv()

from kaggle_ai_core.config import DATASETS_DIR, PUBLISHED_REFS_PATH, ensure_directories, get_settings  # noqa: E402
from kaggle_ai_core.dataset_analyzer import DatasetAnalyzer  # noqa: E402
from kaggle_ai_core.dataset_selector import DatasetSelector  # noqa: E402
from kaggle_ai_core.kaggle_client import KaggleClient  # noqa: E402
from kaggle_ai_core.nemotron_client import NemotronClient  # noqa: E402

from app.model_task_planner import ModelTaskPlanner  # noqa: E402
from app.training_script_generator import TrainingScriptGenerator  # noqa: E402
from pathlib import Path
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

    planner = ModelTaskPlanner(nemotron_client)
    chosen_plan, chosen_profile = None, None
    for profile in profiles:
        plan = planner.generate_plan(profile)
        if plan.trainable:
            chosen_plan, chosen_profile = plan, profile
            break

    if chosen_plan is None:
        print("No trainable candidate found.")
        return

    print(f"\nPlan: {chosen_plan.algorithm} on {chosen_plan.target_column} ({chosen_profile.ref})\n")

    dataset_dir = DATASETS_DIR / chosen_profile.ref.replace("/", "__")
    output_dir = Path("models_out") / chosen_profile.ref.replace("/", "__")
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = TrainingScriptGenerator(nemotron_client)
    script = generator.generate(chosen_plan, chosen_profile, dataset_dir, output_dir)

    print(f"\n{'='*70}\nAPPROACH NOTES\n{'='*70}")
    print(script.approach_notes)
    print(f"\n{'='*70}\nGENERATED SCRIPT ({len(script.code.splitlines())} lines)\n{'='*70}")
    print(script.code)


if __name__ == "__main__":
    main()