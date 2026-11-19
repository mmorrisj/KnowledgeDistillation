import pytest

from kd_course.results import format_results, load_results, save_result


def test_saved_results_can_be_loaded_back(tmp_path):
    save_result("teacher", {"accuracy": 0.71}, predictions=[0, None, 2], results_dir=tmp_path)
    loaded = load_results(results_dir=tmp_path)
    assert loaded["teacher"]["metrics"] == {"accuracy": 0.71}
    assert loaded["teacher"]["predictions"] == [0, None, 2]


def test_saving_twice_replaces_the_earlier_result(tmp_path):
    save_result("teacher", {"accuracy": 0.5}, predictions=[], results_dir=tmp_path)
    save_result("teacher", {"accuracy": 0.7}, predictions=[], results_dir=tmp_path)
    assert load_results(results_dir=tmp_path)["teacher"]["metrics"]["accuracy"] == 0.7


def test_loading_from_a_missing_folder_gives_nothing(tmp_path):
    assert load_results(results_dir=tmp_path / "none") == {}


@pytest.mark.parametrize("name", ["../escape", "a/b", "", "with space"])
def test_names_that_are_not_simple_are_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="name"):
        save_result(name, {}, predictions=[], results_dir=tmp_path)


def test_format_results_shows_each_model_with_its_interval():
    table = format_results(
        {
            "teacher": {
                "metrics": {"accuracy": 0.712, "accuracy_ci": [0.684, 0.74], "macro_f1": 0.69}
            },
            "baseline": {"metrics": {"accuracy": 0.55, "macro_f1": 0.52}},
        }
    )
    assert "teacher" in table and "baseline" in table
    assert "71.2%" in table and "68.4" in table and "74.0" in table
    assert "55.0%" in table


def test_default_results_dir_is_at_the_top_of_the_repo(tmp_path, monkeypatch):
    from kd_course.results import default_results_dir

    (tmp_path / "pyproject.toml").write_text("")
    nested = tmp_path / "labs" / "02_baselines"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.delenv("KD_COURSE_RESULTS_DIR", raising=False)
    assert default_results_dir() == tmp_path / "results"


def test_default_results_dir_can_be_set_by_environment(tmp_path, monkeypatch):
    from kd_course.results import default_results_dir

    monkeypatch.setenv("KD_COURSE_RESULTS_DIR", str(tmp_path / "elsewhere"))
    assert default_results_dir() == tmp_path / "elsewhere"


def test_default_results_dir_falls_back_to_the_current_folder(tmp_path, monkeypatch):
    from kd_course.results import default_results_dir

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KD_COURSE_RESULTS_DIR", raising=False)
    assert default_results_dir() == tmp_path / "results"


def test_two_notebook_folders_share_one_results_folder(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("")
    monkeypatch.delenv("KD_COURSE_RESULTS_DIR", raising=False)
    for folder, name in (("labs/a", "first"), ("labs/b", "second")):
        (tmp_path / folder).mkdir(parents=True)
        monkeypatch.chdir(tmp_path / folder)
        save_result(name, {"accuracy": 0.5}, predictions=[])
    assert set(load_results()) == {"first", "second"}


def test_require_result_returns_a_saved_result(tmp_path):
    from kd_course.results import require_result

    save_result("baseline_gold_616", {"accuracy": 0.6}, predictions=[1], results_dir=tmp_path)
    assert require_result("baseline_gold_616", results_dir=tmp_path)["metrics"]["accuracy"] == 0.6


def test_require_result_explains_what_is_missing(tmp_path):
    from kd_course.results import require_result

    with pytest.raises(LookupError, match="No result named baseline_gold_616.*earlier lab"):
        require_result("baseline_gold_616", results_dir=tmp_path)


def test_load_results_ignores_other_json_files_in_the_folder(tmp_path):
    save_result("teacher", {"accuracy": 0.7}, predictions=[], results_dir=tmp_path)
    (tmp_path / "teacher_scores_pool.json").write_text('{"texts": "abc", "scores": [[-0.1]]}')
    (tmp_path / "notes.json").write_text("[1, 2, 3]")
    assert set(load_results(results_dir=tmp_path)) == {"teacher"}
