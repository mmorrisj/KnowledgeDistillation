"""Load the course's teacher and student models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# One model family, so teacher and student split text into tokens the same way.
TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"
STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"

_DTYPES = ("bfloat16", "float16", "float32")


def resolve_model_path(model_id: str, model_dir: str | None = None) -> str:
    """Use a pre-downloaded copy if one exists, otherwise the Hugging Face id.

    Looks for `<model_dir>/<model name>`, where `model_dir` defaults to the
    `KD_COURSE_MODEL_DIR` environment variable. This lets the labs run on
    machines that cannot reach Hugging Face.
    """
    model_dir = model_dir or os.environ.get("KD_COURSE_MODEL_DIR")
    if model_dir:
        local = Path(model_dir) / model_id.split("/")[-1]
        if local.is_dir():
            return str(local)
    return model_id


def load_causal_lm(
    model_id: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    model_dir: str | None = None,
) -> tuple[Any, Any]:
    """Load a text-generation model and its tokenizer, ready for inference."""
    if dtype not in _DTYPES:
        raise ValueError(f"Unknown dtype '{dtype}'. Use one of: {', '.join(_DTYPES)}.")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = resolve_model_path(model_id, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype))
    return model.to(device).eval(), tokenizer


def count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def load_causal_lm_4bit(model_id: str, model_dir: str | None = None) -> tuple[Any, Any]:
    """Load a model with its weights squeezed to 4 bits each, ready for QLoRA.

    Needs a GPU and the bitsandbytes library. The frozen model takes roughly a third
    of the memory of the 16-bit version; adapters added afterwards train at full precision.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    path = resolve_model_path(model_id, model_dir)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=quantization, device_map={"": 0}
    )
    return model, tokenizer
