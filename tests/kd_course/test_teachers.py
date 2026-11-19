import json

import pytest

httpx = pytest.importorskip("httpx")

from kd_course.teachers import ApiTeacher, TeacherError, classify_with_teacher  # noqa: E402


class FakeTeacher:
    """Replies from a fixed table; records the prompts it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompts, max_new_tokens=16):
        self.prompts.extend(prompts)
        return [self.replies.pop(0) for _ in prompts]


def test_classify_with_teacher_parses_each_reply():
    teacher = FakeTeacher(["cat", "I am not sure", "DOG."])
    assert classify_with_teacher(teacher, ["a", "b", "c"], ["cat", "dog"]) == [0, None, 1]


def test_classify_with_teacher_sends_one_prompt_per_text_with_examples():
    teacher = FakeTeacher(["cat", "dog"])
    classify_with_teacher(teacher, ["first", "second"], ["cat", "dog"], examples=[("meow", "cat")])
    assert len(teacher.prompts) == 2
    assert "first" in teacher.prompts[0] and "meow" in teacher.prompts[0]


def api(handler, **kwargs):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ApiTeacher(
        base_url="https://api.test/v1",
        api_key="secret-key",
        model="m",
        client=client,
        sleep=lambda seconds: None,
        **kwargs,
    )


def ok(content, prompt_tokens=10, completion_tokens=2):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        },
    )


def test_api_teacher_sends_an_openai_style_request():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return ok("cat")

    assert api(handler).complete(["hello"], max_new_tokens=8) == ["cat"]
    assert seen["url"] == "https://api.test/v1/chat/completions"
    assert seen["auth"] == "Bearer secret-key"
    assert seen["body"]["model"] == "m"
    assert seen["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert seen["body"]["max_tokens"] == 8
    assert seen["body"]["temperature"] == 0


def test_api_teacher_adds_up_usage():
    teacher = api(lambda request: ok("x", 100, 5))
    teacher.complete(["a", "b", "c"])
    assert teacher.usage == {"requests": 3, "input_tokens": 300, "output_tokens": 15}


def test_api_teacher_reports_cost_from_usage():
    teacher = api(lambda request: ok("x", 1_000_000, 500_000))
    teacher.complete(["a"])
    assert teacher.cost(usd_per_mtok_in=3.0, usd_per_mtok_out=15.0) == pytest.approx(10.5)


def test_api_teacher_retries_a_rate_limit_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={}) if len(calls) < 3 else ok("cat")

    assert api(handler).complete(["hello"]) == ["cat"]
    assert len(calls) == 3


def test_api_teacher_gives_up_after_repeated_server_errors():
    teacher = api(lambda request: httpx.Response(500, json={}), max_retries=2)
    with pytest.raises(TeacherError, match="500"):
        teacher.complete(["hello"])


def test_api_teacher_does_not_retry_a_bad_key():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": "bad key"})

    with pytest.raises(TeacherError, match="401"):
        api(handler).complete(["hello"])
    assert len(calls) == 1


def test_api_teacher_errors_never_include_the_key():
    teacher = api(lambda request: httpx.Response(401, json={"error": "secret-key is wrong"}))
    with pytest.raises(TeacherError) as error:
        teacher.complete(["hello"])
    assert "secret-key" not in str(error.value)


def test_api_teacher_treats_an_empty_reply_as_empty_text():
    teacher = api(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": None}}]})
    )
    assert teacher.complete(["hello"]) == [""]


def test_api_teacher_reads_settings_from_the_environment(monkeypatch):
    monkeypatch.setenv("KD_COURSE_API_BASE", "https://example.test/v1/")
    monkeypatch.setenv("KD_COURSE_API_KEY", "k")
    monkeypatch.setenv("KD_COURSE_API_MODEL", "some-model")
    teacher = ApiTeacher.from_env()
    assert teacher.base_url == "https://example.test/v1"
    assert teacher.model == "some-model"


def test_api_teacher_from_env_says_what_is_missing(monkeypatch):
    for name in ("KD_COURSE_API_BASE", "KD_COURSE_API_KEY", "KD_COURSE_API_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(TeacherError, match="KD_COURSE_API_BASE"):
        ApiTeacher.from_env()


def test_local_teacher_returns_one_reply_per_prompt(tiny_lm):
    from kd_course.teachers import LocalTeacher

    model, tokenizer = tiny_lm
    teacher = LocalTeacher(model, tokenizer, batch_size=2)
    replies = teacher.complete(["the cat sat", "a dog", "the mat"], max_new_tokens=4)
    assert len(replies) == 3
    assert all(isinstance(reply, str) for reply in replies)
    assert tokenizer.padding_side == "right"  # left padding is used inside and then restored
