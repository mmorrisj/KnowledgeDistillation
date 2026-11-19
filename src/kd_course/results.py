"""Save each model's scores and predictions so later modules can compare against them."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_DIR = "results"
_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def save_result(
    name: str,
    metrics: dict[str, Any],
    predictions: list[int | None],
    results_dir: str | Path = DEFAULT_DIR,
) -> Path:
    """Write `<results_dir>/<name>.json`, replacing any earlier result with that name."""
    if not _NAME.match(name):
        raise ValueError(
            f"Result name '{name}' must use only letters, digits, hyphens and underscores."
        )
    folder = Path(results_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps({"metrics": metrics, "predictions": predictions}, indent=2))
    return path


def load_results(results_dir: str | Path = DEFAULT_DIR) -> dict[str, dict[str, Any]]:
    """Every saved result, keyed by name."""
    folder = Path(results_dir)
    if not folder.is_dir():
        return {}
    return {path.stem: json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))}


def format_results(results: dict[str, dict[str, Any]]) -> str:
    """A table of accuracy and macro-F1, with the accuracy interval where one was saved."""
    lines = [f"{'Model':<24}{'Accuracy':>10}{'95% interval':>18}{'Macro-F1':>10}"]
    for name, result in results.items():
        metrics = result["metrics"]
        interval = metrics.get("accuracy_ci")
        interval_text = f"{interval[0] * 100:.1f} to {interval[1] * 100:.1f}" if interval else ""
        macro = f"{metrics['macro_f1'] * 100:.1f}%" if "macro_f1" in metrics else ""
        lines.append(f"{name:<24}{metrics['accuracy'] * 100:>9.1f}%{interval_text:>18}{macro:>10}")
    return "\n".join(lines)
