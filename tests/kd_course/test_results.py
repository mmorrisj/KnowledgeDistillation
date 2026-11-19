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
