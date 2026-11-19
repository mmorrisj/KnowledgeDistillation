"""Teachers you can only talk to in text: a local model treated as a black box, or an API.

Both have one method, `complete(prompts) -> replies`, so the rest of the course code
does not care which one it is given.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any, Protocol

from kd_course.prompted import build_classification_prompt, parse_label


class TeacherError(RuntimeError):
    """A teacher could not be reached or refused the request."""


class Teacher(Protocol):
    def complete(self, prompts: list[str], max_new_tokens: int = 16) -> list[str]: ...


class LocalTeacher:
    """A model on this machine, used only through its text replies."""

    def __init__(self, model: Any, tokenizer: Any, batch_size: int = 16):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def _as_input(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        return prompt

    def complete(self, prompts: list[str], max_new_tokens: int = 16) -> list[str]:
        import torch

        tokenizer = self.tokenizer
        pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        original_side = tokenizer.padding_side
        tokenizer.padding_side = "left"  # generation continues from the right-hand end
        replies: list[str] = []
        try:
            for start in range(0, len(prompts), self.batch_size):
                batch = [self._as_input(p) for p in prompts[start : start + self.batch_size]]
                inputs = tokenizer(batch, return_tensors="pt", padding=True).to(self.model.device)
                with torch.inference_mode():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                    )
                new_tokens = output[:, inputs["input_ids"].shape[1] :]
                replies.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        finally:
            tokenizer.padding_side = original_side
        return replies


class ApiTeacher:
    """A teacher behind an OpenAI-style chat completions API.

    Keeps a running count of requests and tokens so you can see what a labelling run cost.
    Check the provider's terms before training on its outputs (lesson 1.5).
    """

    RETRY_STATUSES = (429, 500, 502, 503, 504)

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        client: Any = None,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._max_retries = max_retries
        self._sleep = sleep
        if client is None:
            import httpx

            client = httpx.Client(timeout=timeout)
        self._client = client
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

    @classmethod
    def from_env(cls) -> ApiTeacher:
        """Read KD_COURSE_API_BASE, KD_COURSE_API_KEY and KD_COURSE_API_MODEL."""
        names = ("KD_COURSE_API_BASE", "KD_COURSE_API_KEY", "KD_COURSE_API_MODEL")
        missing = [name for name in names if not os.environ.get(name)]
        if missing:
            raise TeacherError(f"Set these environment variables first: {', '.join(missing)}.")
        base, key, model = (os.environ[name] for name in names)
        return cls(base_url=base, api_key=key, model=model)

    def _one(self, prompt: str, max_new_tokens: int) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        for attempt in range(self._max_retries + 1):
            response = self._client.post(
                f"{self.base_url}/chat/completions", json=body, headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage") or {}
                self.usage["requests"] += 1
                self.usage["input_tokens"] += usage.get("prompt_tokens", 0)
                self.usage["output_tokens"] += usage.get("completion_tokens", 0)
                return data["choices"][0]["message"].get("content") or ""
            if response.status_code not in self.RETRY_STATUSES or attempt == self._max_retries:
                # The response body is left out on purpose: it can echo back the key.
                raise TeacherError(
                    f"The teacher API returned status {response.status_code} "
                    f"for model {self.model}."
                )
            self._sleep(min(30.0, 2.0**attempt))
        raise TeacherError("The teacher API could not be reached.")  # pragma: no cover

    def complete(self, prompts: list[str], max_new_tokens: int = 16) -> list[str]:
        return [self._one(prompt, max_new_tokens) for prompt in prompts]

    def cost(self, usd_per_mtok_in: float, usd_per_mtok_out: float) -> float:
        """What the calls so far cost, at the prices you give (per million tokens)."""
        return (
            self.usage["input_tokens"] * usd_per_mtok_in
            + self.usage["output_tokens"] * usd_per_mtok_out
        ) / 1_000_000


def classify_with_teacher(
    teacher: Teacher,
    texts: list[str],
    label_names: list[str],
    examples: list[tuple[str, str]] | None = None,
    max_new_tokens: int = 16,
) -> list[int | None]:
    """Ask a teacher for the type of each text. A reply naming no single type gives None."""
    prompts = [build_classification_prompt(text, label_names, examples) for text in texts]
    replies = teacher.complete(prompts, max_new_tokens=max_new_tokens)
    return [parse_label(reply, label_names) for reply in replies]
