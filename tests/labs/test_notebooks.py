"""Structural checks for lab notebooks. They are not executed here: that needs a GPU."""

import ast
import importlib
from pathlib import Path

import nbformat
import pytest

LABS = Path(__file__).resolve().parents[2] / "labs"
NOTEBOOKS = sorted(LABS.rglob("*.ipynb"))


def code_cells(path: Path) -> list[str]:
    notebook = nbformat.read(path, as_version=4)
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def strip_magics(source: str) -> str:
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
    )


def test_at_least_one_notebook_found():
    assert NOTEBOOKS, f"No notebooks found under {LABS}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid(path):
    nbformat.validate(nbformat.read(path, as_version=4))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_outputs(path):
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [], "Clear outputs before committing"
            assert cell.execution_count is None, "Clear outputs before committing"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_code_cells_parse(path):
    for source in code_cells(path):
        ast.parse(strip_magics(source))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_imports_from_kd_course_exist(path):
    pytest.importorskip("torch")
    found = 0
    for source in code_cells(path):
        for node in ast.walk(ast.parse(strip_magics(source))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("kd_course"):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{node.module} has no '{alias.name}'"
                    found += 1
    assert found, "A lab notebook should use the shared kd_course package"
