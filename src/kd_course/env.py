"""Check that a machine is ready to run the course labs."""

from __future__ import annotations

import platform
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GpuInfo:
    name: str
    memory_gb: float
    supports_bf16: bool


@dataclass(frozen=True)
class EnvReport:
    python: str
    torch: str | None
    gpu: GpuInfo | None
    free_disk_gb: float
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        gpu = f"{self.gpu.name}, {self.gpu.memory_gb:.1f} GB" if self.gpu else "none found"
        lines = [
            f"Python     {self.python}",
            f"PyTorch    {self.torch or 'not installed'}",
            f"GPU        {gpu}",
            f"Free disk  {self.free_disk_gb:.0f} GB",
        ]
        if self.ok:
            lines.append("Ready: this machine can run the course labs.")
        else:
            lines.append("Not ready:")
            lines.extend(f"  - {problem}" for problem in self.problems)
        return "\n".join(lines)


def _torch_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.__version__


def _probe_gpu() -> GpuInfo | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(0)
    return GpuInfo(
        name=properties.name,
        memory_gb=properties.total_memory / 1024**3,
        supports_bf16=torch.cuda.is_bf16_supported(),
    )


def check_environment(
    min_gpu_gb: float = 22.0,
    min_disk_gb: float = 40.0,
    probe: Callable[[], GpuInfo | None] | None = None,
    disk_path: str = ".",
) -> EnvReport:
    """Report what this machine has and list anything that would stop a lab from running."""
    torch_version = _torch_version()
    gpu = (probe or _probe_gpu)()
    free_disk_gb = shutil.disk_usage(disk_path).free / 1024**3

    problems: list[str] = []
    if torch_version is None:
        problems.append("PyTorch is not installed. Run `make install-labs` on this machine.")
    if gpu is None:
        problems.append("No GPU was found. The labs need an NVIDIA GPU with CUDA.")
    else:
        if gpu.memory_gb < min_gpu_gb:
            problems.append(
                f"The GPU has {gpu.memory_gb:.1f} GB of memory; the labs need at least "
                f"{min_gpu_gb:.0f} GB (an A10G or larger)."
            )
        if not gpu.supports_bf16:
            problems.append(
                "The GPU does not support bf16, which the labs use for training. "
                "Use an A10G or a newer card."
            )
    if free_disk_gb < min_disk_gb:
        problems.append(
            f"Only {free_disk_gb:.0f} GB of disk is free; the labs need about "
            f"{min_disk_gb:.0f} GB for models and cached data."
        )

    return EnvReport(
        python=platform.python_version(),
        torch=torch_version,
        gpu=gpu,
        free_disk_gb=free_disk_gb,
        problems=problems,
    )
