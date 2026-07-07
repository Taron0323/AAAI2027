from foreact.evaluation.datasets import dataset_status


def test_dataset_status_marks_missing_paths():
    status = dataset_status(
        {
            "tau2_bench": {
                "required_paths": {"root": "${SURELY_MISSING_FOREACT_TEST_ROOT}/tau2"},
                "setup": "set path",
            }
        }
    )
    assert status["tau2_bench"]["available"] is False
    assert status["tau2_bench"]["missing"] == ["root"]
    assert "not vendored" in status["tau2_bench"]["note"]


def test_dataset_status_resolves_relative_paths(tmp_path, monkeypatch):
    marker = tmp_path / "datasets" / "demo" / "READY"
    marker.parent.mkdir(parents=True)
    marker.write_text("ok\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    status = dataset_status(
        {
            "demo": {
                "required_paths": {"marker": "datasets/demo/READY"},
                "setup": "download demo",
            }
        }
    )
    assert status["demo"]["available"] is True
    assert status["demo"]["paths"]["marker"]["path"] == str(marker.resolve())
