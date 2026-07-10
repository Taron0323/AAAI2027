from foreact.environments.plandepth import PlanDepthEnv
from foreact.evaluation.metrics import evaluate_plandepth
from foreact.inference.policies import OracleSFTPolicy, ReActPolicy


def test_plandepth_distinguishes_myopic_and_expert():
    env = PlanDepthEnv(seed=0)
    tasks = env.make_tasks(num_tasks=3, depths=[4], stochastic=False, delayed_deadend=True)
    react = [ReActPolicy().act_plan(task) for task in tasks]
    sft = [OracleSFTPolicy().act_plan(task) for task in tasks]
    react_metrics = evaluate_plandepth(tasks, react)
    sft_metrics = evaluate_plandepth(tasks, sft)
    assert react_metrics["dead_end_rate"] == 1.0
    assert sft_metrics["success_rate"] == 1.0

