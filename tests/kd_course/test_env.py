import importlib
import sys

import pytest

from kd_course import env
from kd_course.env import GpuInfo, check_environment

A10G = GpuInfo(name="NVIDIA A10G", memory_gb=22.5, supports_bf16=True)


@pytest.fixture(autouse=True)
def torch_installed(monkeypatch):
    """Most tests describe a machine with PyTorch, whether or not this one has it."""
    monkeypatch.setattr(env, "_torch_version", lambda: "2.5.0")


def test_ok_on_a10g():
    report = check_environment(probe=lambda: A10G, min_disk_gb=0)
    assert report.ok
    assert report.problems == []
    assert report.gpu == A10G


def test_no_gpu_is_a_problem():
    report = check_environment(probe=lambda: None, min_disk_gb=0)
    assert not report.ok
    assert "No GPU" in report.problems[0]


def test_small_gpu_is_a_problem():
    small = GpuInfo(name="Tesla T4", memory_gb=15.8, supports_bf16=True)
    report = check_environment(probe=lambda: small, min_disk_gb=0)
    assert any("15.8" in p and "22" in p for p in report.problems)


def test_no_bf16_is_a_problem():
    v100 = GpuInfo(name="Tesla V100", memory_gb=32.0, supports_bf16=False)
    report = check_environment(probe=lambda: v100, min_disk_gb=0)
    assert any("bf16" in p for p in report.problems)


def test_low_disk_is_a_problem():
    report = check_environment(probe=lambda: A10G, min_disk_gb=10**6)
    assert any("disk" in p.lower() for p in report.problems)


def test_summary_lists_each_problem_on_its_own_line():
    v100 = GpuInfo(name="Tesla V100", memory_gb=15.8, supports_bf16=False)
    report = check_environment(probe=lambda: v100, min_disk_gb=0)
    summary = report.summary()
    assert len(report.problems) == 2
    for problem in report.problems:
        assert f"\n  - {problem}" in summary


def test_summary_says_ready_when_ok():
    summary = check_environment(probe=lambda: A10G, min_disk_gb=0).summary()
    assert "NVIDIA A10G" in summary
    assert "Ready" in summary


def test_module_imports_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    reloaded = importlib.reload(env)
    report = reloaded.check_environment(min_disk_gb=0)
    assert report.torch is None
    assert any("PyTorch" in p for p in report.problems)
