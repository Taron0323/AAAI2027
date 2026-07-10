from pathlib import Path

from foreact.evaluation.assets import asset_status, asset_summary


def test_asset_status_distinguishes_missing_code_and_dataset(tmp_path: Path):
    manifest = {
        "benchmark_code": {
            "demo": {
                "title": "Demo",
                "local_path": "third_party/benchmarks/demo",
                "required_markers": ["README.md"],
                "url": "https://example.invalid/demo.git",
            }
        },
        "datasets": {
            "demo_data": {
                "title": "Demo Data",
                "local_path": "datasets/demo",
                "required_markers": ["split.jsonl"],
            }
        },
    }
    status = asset_status(manifest, project_root=tmp_path)
    assert status["benchmark_code"]["demo"]["available"] is False
    assert status["datasets"]["demo_data"]["available"] is False

    code_dir = tmp_path / "third_party" / "benchmarks" / "demo"
    code_dir.mkdir(parents=True)
    (code_dir / "README.md").write_text("demo\n", encoding="utf-8")
    status = asset_status(manifest, project_root=tmp_path)
    assert status["benchmark_code"]["demo"]["available"] is True
    assert asset_summary(status)["benchmark_code"] == {"available": 1, "missing": 0, "total": 1}
