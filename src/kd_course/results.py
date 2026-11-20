"""Save each model's scores and predictions so later modules can compare against them."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def default_results_dir() -> Path:
    """One shared folder for every lab.

    Uses `KD_COURSE_RESULTS_DIR` if set, otherwise `results` at the top of the repo.

    The repo top is found by walking up from the current folder to `pyproject.toml`.
    Outside a repo, `results` in the current folder is used.
    """
    configured = os.environ.get("KD_COURSE_RESULTS_DIR")
    if configured:
        return Path(configured)
    here = Path.cwd()
    for folder in (here, *here.parents):
        if (folder / "pyproject.toml").is_file():
            return folder / "results"
    return here / "results"


_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def save_result(
    name: str,
    metrics: dict[str, Any],
    predictions: list[Any],
    results_dir: str | Path | None = None,
) -> Path:
    """Write `<results_dir>/<name>.json`, replacing any earlier result with that name."""
    if not _NAME.match(name):
        raise ValueError(
            f"Result name '{name}' must use only letters, digits, hyphens and underscores."
        )
    folder = Path(results_dir) if results_dir else default_results_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps({"metrics": metrics, "predictions": predictions}, indent=2))
    return path


def load_results(results_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Every saved result, keyed by name."""
    folder = Path(results_dir) if results_dir else default_results_dir()
    if not folder.is_dir():
        return {}
    results = {}
    for path in sorted(folder.glob("*.json")):
        content = json.loads(path.read_text())
        # Other files, such as saved teacher scores, can live in the same folder.
        if isinstance(content, dict) and isinstance(content.get("metrics"), dict):
            results[path.stem] = content
    return results


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


def require_result(name: str, results_dir: str | Path | None = None) -> dict[str, Any]:
    """A result an earlier lab should have saved, or a clear error saying which one is missing."""
    results = load_results(results_dir)
    if name not in results:
        folder = Path(results_dir) if results_dir else default_results_dir()
        raise LookupError(
            f"No result named {name} in {folder}. Run the earlier lab that produces it first."
        )
    return results[name]
