"""
companion_notebook_builder.py

Builds a small, deterministic "starter" notebook demonstrating how to
load and use a published Kaggle Model -- satisfies Kaggle's "Publish a
notebook" completeness recommendation. Not LLM-generated: loading a
model and calling .predict() on sample rows is a fixed, mechanical
pattern that doesn't need Nemotron's involvement or risk hallucination.

Like the notebook-agent's setup cell, this searches /kaggle/input rather
than hardcoding a mount path, since Kaggle's model/dataset mount
conventions vary and shouldn't be assumed.
"""

from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from app.model_task_planner import ModelPlan

NOTEBOOK_KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}
NOTEBOOK_LANGUAGE_INFO = {"name": "python", "version": "3.12"}


def build_companion_notebook(plan: ModelPlan, primary_csv: str) -> nbformat.NotebookNode:
    """Build a minimal usage-demo notebook for a published model."""
    nb = new_notebook()
    nb.metadata["kernelspec"] = NOTEBOOK_KERNELSPEC
    nb.metadata["language_info"] = NOTEBOOK_LANGUAGE_INFO

    nb.cells.append(new_markdown_cell(
        f"# {plan.model_title}\n\n"
        f"Quick-start example: load the published model and run predictions on a few sample rows.\n\n"
        f"**Task:** {plan.problem_type} | **Target:** `{plan.target_column}` | **Algorithm:** `{plan.algorithm}`"
    ))

    setup_code = (
        "import cloudpickle\n"
        "import pandas as pd\n"
        "from pathlib import Path\n\n"
        "# Search /kaggle/input for the model and dataset files rather than assuming a\n"
        "# specific mount path -- Kaggle's mount conventions vary between personal and\n"
        "# organization-owned datasets/models.\n"
        "_kaggle_input = Path(\"/kaggle/input\")\n\n"
        "_model_matches = list(_kaggle_input.rglob(\"model.joblib\"))\n"
        "if not _model_matches:\n"
        "    raise FileNotFoundError(\"model.joblib not found under /kaggle/input -- ensure this model is attached to the notebook.\")\n"
        "with open(_model_matches[0], \"rb\") as _f:\n"
        "    model = cloudpickle.load(_f)\n"
        "print(f\"Loaded model from: {_model_matches[0]}\")\n\n"
        f'_data_matches = list(_kaggle_input.rglob("{primary_csv}"))\n'
        f'if not _data_matches:\n'
        f'    raise FileNotFoundError("{primary_csv} not found under /kaggle/input -- ensure the source dataset is attached.")\n'
        "df = pd.read_csv(_data_matches[0])\n"
        "print(f\"Loaded data from: {_data_matches[0]} ({len(df)} rows)\")\n"
        "df.head()"
    )
    nb.cells.append(new_code_cell(setup_code))

    nb.cells.append(new_markdown_cell("## Run predictions on a small sample\n\nUsing the fitted pipeline's `.predict()` directly on raw sample rows (preprocessing is included in the saved pipeline)."))

    predict_code = (
        f"sample = df.drop(columns=[\"{plan.target_column}\"], errors=\"ignore\").sample(\n"
        f"    n=min(10, len(df)), random_state=42\n"
        f")\n"
        f"predictions = model.predict(sample)\n\n"
        f"results = sample.copy()\n"
        f'results["predicted_{plan.target_column}"] = predictions\n'
        f"results"
    )
    nb.cells.append(new_code_cell(predict_code))

    nb.cells.append(new_markdown_cell(
        "## Notes\n\n"
        f"{plan.model_card_summary}\n\n"
        "This notebook was generated automatically as a usage example. "
        "See the Model Card for full training methodology and evaluation metrics."
    ))

    return nb