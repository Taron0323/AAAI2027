from pathlib import Path

from foreact.analysis.mechanism import (
    effective_lookahead_depth,
    eld_curve_rows,
    granularity_diagnostic_rows,
    plandepth_boundary_rows,
)
from foreact.baselines.registry import registry
from foreact.evaluation.assets import asset_audit, scan_model_weight_files


def test_effective_lookahead_depth_contract():
    curve = {1: 0.8, 2: 0.7, 3: 0.1, 4: 0.01}
    assert effective_lookahead_depth(curve, relative_threshold=0.25) == 2
    rows = eld_curve_rows("react_sft_smoke", curve, figure_role="Fig2_pilot")
    assert rows[0]["axis_depth_min"] == 1
    assert rows[0]["axis_depth_max"] == 4
    assert rows[0]["effective_lookahead_depth"] == 2


def test_plandepth_boundary_and_granularity_rows():
    rows = plandepth_boundary_rows(
        {"foreact": {"sr_by_depth": {"2": 1.0, "4": 0.5, "8": 0.0}}},
        {"foreact": 4},
        trained_horizon=4,
    )
    assert rows[-1]["beyond_trained_horizon"] is True
    assert rows[-1]["within_eld"] is False
    modes = {row["zeta_mode"] for row in granularity_diagnostic_rows()}
    assert {"token", "fsp_summary", "type", "type_arg", "vq_mock"}.issubset(modes)


def test_registry_contains_v3_hypotheses_and_guardrails():
    data = registry()
    hypothesis_keys = {item["key"] for item in data["hypotheses"]}
    assert hypothesis_keys == {"H1", "H2", "H3", "H4", "H5"}
    alignment = {item["contribution"]: item for item in data["contribution_alignment"]}
    assert "token-level" in alignment["C1"]["paper_guardrail"]
    assert "KL(sg" in alignment["C2"]["paper_guardrail"]


def test_asset_audit_rejects_model_weights(tmp_path: Path):
    manifest = {"benchmark_code": {}}
    assert asset_audit(manifest, project_root=tmp_path)["no_model_weights"] is True
    weight = tmp_path / "third_party" / "bad" / "model.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_text("not a real model", encoding="utf-8")
    assert scan_model_weight_files(tmp_path) == [str(weight)]
    audit = asset_audit(manifest, project_root=tmp_path)
    assert audit["no_model_weights"] is False
    assert audit["model_weight_files"] == [str(weight)]
