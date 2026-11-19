import pytest

from kd_course.models import STUDENT_ID, TEACHER_ID, resolve_model_path


def test_default_models_share_a_family():
    assert TEACHER_ID == "Qwen/Qwen2.5-7B-Instruct"
    assert STUDENT_ID == "Qwen/Qwen2.5-0.5B-Instruct"


def test_resolve_prefers_local_folder(tmp_path):
    local = tmp_path / "Qwen2.5-0.5B-Instruct"
    local.mkdir()
    assert resolve_model_path(STUDENT_ID, model_dir=str(tmp_path)) == str(local)


def test_resolve_falls_back_to_hub_id(tmp_path):
    assert resolve_model_path(STUDENT_ID, model_dir=str(tmp_path)) == STUDENT_ID


def test_resolve_reads_env_var(monkeypatch, tmp_path):
    (tmp_path / "Qwen2.5-7B-Instruct").mkdir()
    monkeypatch.setenv("KD_COURSE_MODEL_DIR", str(tmp_path))
    assert resolve_model_path(TEACHER_ID) == str(tmp_path / "Qwen2.5-7B-Instruct")


def test_resolve_without_env_var_returns_hub_id(monkeypatch):
    monkeypatch.delenv("KD_COURSE_MODEL_DIR", raising=False)
    assert resolve_model_path(TEACHER_ID) == TEACHER_ID


def test_count_parameters_matches_manual_sum(tiny_lm):
    from kd_course.models import count_parameters

    model, _ = tiny_lm
    assert count_parameters(model) == sum(p.numel() for p in model.parameters())


def test_load_causal_lm_from_local_folder(tiny_lm, tmp_path):
    from kd_course.models import load_causal_lm

    model, tokenizer = tiny_lm
    folder = tmp_path / "tiny"
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    loaded, loaded_tokenizer = load_causal_lm(
        "org/tiny", device="cpu", dtype="float32", model_dir=str(tmp_path)
    )
    assert not loaded.training
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size


def test_load_causal_lm_rejects_unknown_dtype(tiny_lm):
    from kd_course.models import load_causal_lm

    with pytest.raises(ValueError, match="dtype"):
        load_causal_lm("org/tiny", device="cpu", dtype="float7")
