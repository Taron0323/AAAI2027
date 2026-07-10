from foreact.analysis.visualize import write_case_heatmap_svg, write_depth_svg, write_efficiency_svg


def test_visualization_exports(tmp_path):
    write_depth_svg(tmp_path / "depth.svg", {"react": {"2": 0.5, "4": 0.25}})
    write_efficiency_svg(tmp_path / "eff.svg", {"react": {"success_rate": 0.5, "extra_forward_per_task": 0}})
    write_case_heatmap_svg(
        tmp_path / "case.svg",
        [
            {
                "action": "verify target=x",
                "forecast": [{"depth": 1, "dead_end_mass": 0.2}, {"depth": 2, "dead_end_mass": 0.7}],
            }
        ],
    )
    assert (tmp_path / "depth.svg").read_text(encoding="utf-8").startswith("<svg")
    assert "Success vs. Overhead" in (tmp_path / "eff.svg").read_text(encoding="utf-8")
    assert "Heatmap" in (tmp_path / "case.svg").read_text(encoding="utf-8")
