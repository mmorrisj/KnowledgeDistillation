import pytest

from kd_course.serving.cascade import cascade_stats, sweep_thresholds
from kd_course.serving.loadtest import cost_per_1k, run_load_test, summarize_latencies


def test_summarize_latencies_reports_percentiles_in_milliseconds():
    latencies = [i / 1000 for i in range(1, 101)]  # 1 ms .. 100 ms
    summary = summarize_latencies(latencies)
    assert summary["p50_ms"] == pytest.approx(50.5)
    assert summary["p95_ms"] == pytest.approx(95.05)
    assert summary["mean_ms"] == pytest.approx(50.5)


def test_summarize_latencies_rejects_an_empty_list():
    with pytest.raises(ValueError, match="empty"):
        summarize_latencies([])


def test_run_load_test_calls_once_per_payload_and_counts_items():
    seen = []

    def call(payload):
        seen.append(payload)
        return len(payload)

    result = run_load_test(call, [["a"], ["b", "c"], ["d"]], concurrency=2)
    assert sorted(seen) == [["a"], ["b", "c"], ["d"]]
    assert result.requests == 3
    assert result.items == 4
    assert len(result.latencies) == 3
    assert result.requests_per_second > 0
    assert result.items_per_second >= result.requests_per_second
    assert result.errors == 0


def test_run_load_test_counts_failures_and_carries_on():
    def call(payload):
        if payload == ["bad"]:
            raise RuntimeError("no")
        return 1

    result = run_load_test(call, [["a"], ["bad"], ["c"]], concurrency=1)
    assert result.errors == 1
    assert result.requests == 3
    assert len(result.latencies) == 2


def test_cost_per_1k():
    # $1.62 an hour, 100 items a second -> 360,000 items an hour -> $0.0045 per 1,000
    assert cost_per_1k(1.62, 100) == pytest.approx(0.0045)


def test_cost_per_1k_rejects_zero_throughput():
    with pytest.raises(ValueError, match="above zero"):
        cost_per_1k(1.62, 0)


CONF = [0.95, 0.90, 0.60, 0.55, 0.30]
STUDENT = [True, True, True, False, False]
TEACHER = [True, True, False, True, True]


def test_cascade_with_threshold_zero_uses_only_the_student():
    stats = cascade_stats(CONF, STUDENT, TEACHER, threshold=0.0)
    assert stats["sent_to_teacher"] == 0.0
    assert stats["accuracy"] == pytest.approx(3 / 5)


def test_cascade_with_threshold_above_one_uses_only_the_teacher():
    stats = cascade_stats(CONF, STUDENT, TEACHER, threshold=1.01)
    assert stats["sent_to_teacher"] == 1.0
    assert stats["accuracy"] == pytest.approx(4 / 5)


def test_cascade_sends_only_low_confidence_messages_to_the_teacher():
    # below 0.58: the last two go to the teacher (both right); the first three stay (all right)
    stats = cascade_stats(CONF, STUDENT, TEACHER, threshold=0.58)
    assert stats["sent_to_teacher"] == pytest.approx(2 / 5)
    assert stats["accuracy"] == 1.0


def test_cascade_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length"):
        cascade_stats([0.5], [True, False], [True], 0.5)


def test_sweep_returns_one_row_per_threshold_in_order():
    rows = sweep_thresholds(CONF, STUDENT, TEACHER, [0.0, 0.58, 1.01])
    assert [row["threshold"] for row in rows] == [0.0, 0.58, 1.01]
    assert rows[1]["accuracy"] == 1.0


def test_correct_flags_compares_predictions_with_the_truth():
    from kd_course.serving.cascade import correct_flags

    assert correct_flags([1, 0, None, 2], [1, 1, 0, 2]) == [True, False, False, True]


def test_correct_flags_rejects_mismatched_lengths():
    from kd_course.serving.cascade import correct_flags

    with pytest.raises(ValueError, match="length"):
        correct_flags([1], [1, 2])
