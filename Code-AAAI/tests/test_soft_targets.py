from foreact.data.soft_targets import aggregate_future_targets
from foreact.types import Action
from foreact.zeta import SketchMapper, build_schema_from_actions


def test_aggregate_future_targets_soft_distribution():
    a = Action(text="move target=a", tool="move", args={"target": "a"})
    b = Action(text="query_constraint target=a", tool="query_constraint", args={"target": "a"})
    schema = build_schema_from_actions([a, b])
    mapper = SketchMapper(schema)
    targets = aggregate_future_targets([[a], [b], [b]], mapper, horizon=1)
    assert targets[0].distribution["query_constraint::target"] == 2 / 3
    assert 0.0 < targets[0].entropy <= 1.0
    assert 0.0 <= targets[0].branch_weight < 1.0

