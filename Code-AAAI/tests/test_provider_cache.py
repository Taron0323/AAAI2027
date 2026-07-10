from foreact.teachers.providers import CachedAPIProvider
from foreact.types import Action, TrajectoryPrefix


def test_cached_provider_roundtrip(tmp_path):
    provider = CachedAPIProvider("mock-model", "MISSING_KEY_FOR_TEST", tmp_path)
    prefix = TrajectoryPrefix(
        task_id="task",
        goal="goal",
        actions=[],
        observations=[],
        next_action_index=0,
        success=True,
    )
    expected = [[Action(text="move target=x", tool="move", args={"target": "x"})]]
    provider.write_cache(prefix, horizon=1, k=1, continuations=expected)
    actual = provider.continue_actions(prefix, horizon=1, k=1)
    assert actual[0][0].text == "move target=x"
