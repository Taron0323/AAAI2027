"""Schema-derived action sketch mapping zeta."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Dict, Iterable, List, Mapping, Sequence

from foreact.types import Action


def primary_arg(args: Mapping[str, str]) -> str:
    for key in ("target", "file", "object", "id", "path", "slot"):
        if key in args and args[key]:
            return key
    if args:
        return sorted(args.keys())[0]
    return "none"


def sketch_for_action(action: Action, mode: str = "type_arg") -> str:
    if mode == "token":
        return action.text
    if mode == "type":
        return action.tool
    if mode == "fsp_summary":
        return f"bag::{action.tool}"
    if mode == "vq_mock":
        digest = hashlib.sha256(action.text.encode("utf-8")).hexdigest()
        return f"vq::{int(digest[:8], 16) % 512}"
    if mode != "type_arg":
        raise ValueError(f"unknown sketch mode: {mode}")
    return f"{action.tool}::{primary_arg(action.args)}"


def build_schema_from_actions(actions: Iterable[Action], mode: str = "type_arg") -> Dict[str, object]:
    counter = Counter(sketch_for_action(action, mode=mode) for action in actions)
    sketches = sorted(counter)
    return {
        "mode": mode,
        "size": len(sketches),
        "sketches": sketches,
        "counts": dict(counter),
    }


def build_schema_from_tool_schema(tool_schema: Mapping[str, object], mode: str = "type_arg") -> Dict[str, object]:
    """Build zeta vocabulary from a simple tool-schema document.

    Expected shape:
    {"tools": [{"name": "move", "parameters": {"target": "string"}}]}
    """

    actions: List[Action] = []
    for tool in tool_schema.get("tools", []):  # type: ignore[union-attr]
        if not isinstance(tool, Mapping):
            continue
        name = str(tool.get("name", "unknown"))
        params = tool.get("parameters", {})
        if isinstance(params, Mapping) and params:
            for param in sorted(params):
                actions.append(Action(text=f"{name} {param}=<{param}>", tool=name, args={str(param): f"<{param}>"}))
        else:
            actions.append(Action(text=name, tool=name, args={}))
    return build_schema_from_actions(actions, mode=mode)


@dataclass
class SketchMapper:
    schema: Mapping[str, object]

    @property
    def mode(self) -> str:
        return str(self.schema.get("mode", "type_arg"))

    @property
    def sketches(self) -> Sequence[str]:
        return list(self.schema.get("sketches", []))

    def encode(self, action: Action) -> str:
        sketch = sketch_for_action(action, mode=self.mode)
        if self.sketches and sketch not in self.sketches:
            return "<unk>"
        return sketch

    def encode_many(self, actions: Iterable[Action]) -> List[str]:
        return [self.encode(action) for action in actions]
