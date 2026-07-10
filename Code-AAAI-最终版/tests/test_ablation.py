from foreact.training.ablation import expand_ablation_jobs


def test_expand_ablation_jobs():
    jobs = expand_ablation_jobs(
        {
            "main_controls": [{"key": "react_sft", "description": "baseline"}],
            "sketch_granularity": ["token", "type_arg"],
            "horizon_sweep": [1, 2],
            "rollout_sweep": [1],
            "lambda_sweep": [0.3],
            "mu_sweep": [0.1, 0.0],
        }
    )
    keys = {job["key"] for job in jobs}
    assert "react_sft" in keys
    assert "A_prime_zeta_token" in keys
    assert "D_H2" in keys
    assert "F_lambda0.3_mu0.1" in keys
