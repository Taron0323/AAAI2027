from pathlib import Path

from foreact.cli import run_smoke


def test_cli_smoke_runs(tmp_path):
    config = tmp_path / "smoke.yaml"
    config.write_text(
        """
seed: 1
output_dir: {out}
plandepth:
  num_tasks: 4
  depths: [2, 4]
  stochastic: true
  delayed_deadend: true
  max_steps: 6
zeta:
  mode: type_arg
  schema_out: {out}/schema.json
soft_targets:
  horizon: 3
  rollouts: 3
  prefix_fraction: 1.0
training:
  hidden_dim: 8
  horizon: 3
  lambda_future: 0.3
  mu_consistency: 0.1
  eta_success: 0.05
  steps: 3
  learning_rate: 0.02
inference:
  mode: latent_rerank
  candidates: 3
""".format(out=tmp_path / "out"),
        encoding="utf-8",
    )
    manifest = run_smoke(str(config))
    assert manifest["num_tasks"] == 4
    assert manifest["num_examples"] > 0
    assert "effective_lookahead_depth" in manifest
    assert manifest["v3_contract"]["mode_A_zero_overhead"] is True
    assert (tmp_path / "out" / "efficiency.csv").exists()
    assert (tmp_path / "out" / "fig2_pilot_eld.csv").exists()
    assert (tmp_path / "out" / "fig5_recovery_eld.svg").exists()
    assert (tmp_path / "out" / "success_overhead_pareto.svg").exists()
    assert (tmp_path / "out" / "plandepth_boundary_h5.csv").exists()
    assert (tmp_path / "out" / "granularity_a_prime.csv").exists()
