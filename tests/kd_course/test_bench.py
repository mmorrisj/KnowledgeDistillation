import pytest

from kd_course.bench import BenchResult, compare

PROMPTS = ["the cat sat on", "a dog ran to", "the cat"]


def test_benchmark_reports_speed_and_token_count(tiny_lm):
    from kd_course.bench import benchmark_generation

    model, tokenizer = tiny_lm
    result = benchmark_generation(model, tokenizer, PROMPTS, max_new_tokens=8)
    assert 0 < result.new_tokens <= len(PROMPTS) * 8
    assert result.tokens_per_second > 0
    assert result.latency_p50_s > 0
    assert result.n_params == sum(p.numel() for p in model.parameters())


def test_benchmark_peak_memory_is_none_on_cpu(tiny_lm):
    from kd_course.bench import benchmark_generation

    model, tokenizer = tiny_lm
    assert benchmark_generation(model, tokenizer, PROMPTS, max_new_tokens=4).peak_memory_gb is None


def test_benchmark_rejects_empty_prompts(tiny_lm):
    from kd_course.bench import benchmark_generation

    model, tokenizer = tiny_lm
    with pytest.raises(ValueError, match="prompt"):
        benchmark_generation(model, tokenizer, [])


def test_generate_text_returns_only_the_new_text(tiny_lm):
    from kd_course.bench import generate_text

    model, tokenizer = tiny_lm
    text = generate_text(model, tokenizer, "the cat sat on", max_new_tokens=4)
    assert isinstance(text, str)
    assert not text.startswith("the cat sat on")


def test_compare_ratios():
    teacher = BenchResult(
        n_params=70, tokens_per_second=10, latency_p50_s=2, peak_memory_gb=14, new_tokens=1
    )
    student = BenchResult(
        n_params=5, tokens_per_second=100, latency_p50_s=0.2, peak_memory_gb=1, new_tokens=1
    )
    assert compare(teacher, student) == {"size_ratio": 14.0, "speedup": 10.0, "memory_ratio": 14.0}


def test_compare_omits_memory_when_unknown():
    teacher = BenchResult(
        n_params=70, tokens_per_second=10, latency_p50_s=2, peak_memory_gb=None, new_tokens=1
    )
    student = BenchResult(
        n_params=5, tokens_per_second=100, latency_p50_s=0.2, peak_memory_gb=1, new_tokens=1
    )
    assert "memory_ratio" not in compare(teacher, student)


def test_format_comparison_shows_both_models_and_the_ratios():
    from kd_course.bench import format_comparison

    teacher = BenchResult(
        n_params=7_600_000_000,
        tokens_per_second=30,
        latency_p50_s=2,
        peak_memory_gb=15.2,
        new_tokens=1,
    )
    student = BenchResult(
        n_params=500_000_000,
        tokens_per_second=120,
        latency_p50_s=0.4,
        peak_memory_gb=1.0,
        new_tokens=1,
    )
    text = format_comparison(teacher, student)
    assert "7.60" in text and "0.50" in text
    assert "15x smaller" in text
    assert "4.0x faster" in text
    assert "15x less GPU memory" in text


def test_format_comparison_without_memory_figures():
    from kd_course.bench import format_comparison

    teacher = BenchResult(
        n_params=70, tokens_per_second=10, latency_p50_s=2, peak_memory_gb=None, new_tokens=1
    )
    student = BenchResult(
        n_params=5, tokens_per_second=100, latency_p50_s=0.2, peak_memory_gb=None, new_tokens=1
    )
    text = format_comparison(teacher, student)
    assert "n/a" in text
    assert "GPU memory." not in text
