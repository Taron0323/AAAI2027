from foreact.types import Action
from foreact.zeta import SketchMapper, build_schema_from_actions


def test_schema_mapper_type_arg():
    actions = [
        Action(text="move target=a", tool="move", args={"target": "a"}),
        Action(text="verify target=a", tool="verify", args={"target": "a"}),
    ]
    schema = build_schema_from_actions(actions)
    mapper = SketchMapper(schema)
    assert mapper.encode(actions[0]) == "move::target"
    assert schema["size"] == 2

