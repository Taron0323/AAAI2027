from foreact.zeta.schema import build_schema_from_tool_schema


def test_schema_from_tool_schema():
    schema = build_schema_from_tool_schema(
        {
            "tools": [
                {"name": "inspect", "parameters": {"target": "string"}},
                {"name": "commit", "parameters": {"target": "string"}},
            ]
        }
    )
    assert "inspect::target" in schema["sketches"]
    assert "commit::target" in schema["sketches"]
