"""Measure how fast a model generates text and how much GPU memory it needs."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

from kd_course.models import count_parameters


@dataclass(frozen=True)
class BenchResult:
    n_params: int
    tokens_per_second: float
    latency_p50_s: float
    peak_memory_gb: float | None
    new_tokens: int


def _encode(tokenizer: Any, prompt: str, device: Any) -> Any:
    """Tokenize a prompt, using the model's chat format when it has one."""
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt").to(device)


def _generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> Any:
    """Greedy generation. Returns only the newly generated token ids."""
    import torch

    inputs = _encode(tokenizer, prompt, model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return output[0, inputs["input_ids"].shape[1] :]


def generate_text(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = 64) -> str:
    """The model's reply to one prompt, without the prompt itself."""
    new_ids = _generate(model, tokenizer, prompt, max_new_tokens)
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def benchmark_generation(
    model: Any, tokenizer: Any, prompts: list[str], max_new_tokens: int = 64
) -> BenchResult:
    """Time greedy generation over the prompts, one at a time."""
    if not prompts:
        raise ValueError("Give at least one prompt to benchmark.")
    import torch

    on_gpu = model.device.type == "cuda"
    _generate(model, tokenizer, prompts[0], max_new_tokens=4)  # warm-up, not timed
    if on_gpu:
        torch.cuda.reset_peak_memory_stats(model.device)

    latencies: list[float] = []
    new_tokens = 0
    for prompt in prompts:
        if on_gpu:
            torch.cuda.synchronize(model.device)
        start = time.perf_counter()
        new_ids = _generate(model, tokenizer, prompt, max_new_tokens)
        if on_gpu:
            torch.cuda.synchronize(model.device)
        latencies.append(time.perf_counter() - start)
        new_tokens += int(new_ids.shape[0])

    peak = torch.cuda.max_memory_allocated(model.device) / 1024**3 if on_gpu else None
    return BenchResult(
        n_params=count_parameters(model),
        tokens_per_second=new_tokens / sum(latencies),
        latency_p50_s=statistics.median(latencies),
        peak_memory_gb=peak,
        new_tokens=new_tokens,
    )


def compare(teacher: BenchResult, student: BenchResult) -> dict[str, float]:
    """How many times smaller, faster and lighter the student is."""
    ratios = {
        "size_ratio": teacher.n_params / student.n_params,
        "speedup": student.tokens_per_second / teacher.tokens_per_second,
    }
    if teacher.peak_memory_gb is not None and student.peak_memory_gb is not None:
        ratios["memory_ratio"] = teacher.peak_memory_gb / student.peak_memory_gb
    return ratios
