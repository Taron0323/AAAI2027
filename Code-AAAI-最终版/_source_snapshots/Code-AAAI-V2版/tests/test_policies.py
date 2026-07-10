from foreact.environments.plandepth import PlanDepthEnv
from foreact.evaluation.metrics import evaluate_plandepth
from foreact.inference.policies import PlanAndActPolicy, SearchLookaheadPolicy, TokenMTPPolicy


def test_smoke_baseline_policies_run():
    env = PlanDepthEnv(seed=0)
    tasks = env.make_tasks(num_tasks=2, depths=[4], stochastic=False, delayed_deadend=True)
    for policy in [TokenMTPPolicy(), PlanAndActPolicy(), SearchLookaheadPolicy()]:
        traces = [policy.act_plan(task) for task in tasks]
        metrics = evaluate_plandepth(tasks, traces)
        assert "success_rate" in metrics
        assert metrics["tokens_per_task"] > 0
