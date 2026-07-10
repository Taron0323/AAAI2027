"""Teacher provider abstraction with mock and API-ready implementations."""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Sequence

from foreact.types import Action, TrajectoryPrefix


class TeacherProvider(ABC):
    """Interface for DeepSeek/GPT/self-hosted teacher continuations."""

    @abstractmethod
    def continue_actions(self, prefix: TrajectoryPrefix, horizon: int, k: int) -> List[List[Action]]:
        raise NotImplementedError


class MockTeacherProvider(TeacherProvider):
    """Deterministic local provider for tests and smoke runs."""

    def continue_actions(self, prefix: TrajectoryPrefix, horizon: int, k: int) -> List[List[Action]]:
        continuations: List[List[Action]] = []
        for sample in range(k):
            future = []
            for depth in range(horizon):
                tool = "query_constraint" if (sample + depth) % 4 == 0 else "move"
                target = f"mock_{prefix.next_action_index + depth}"
                future.append(Action(text=f"{tool} target={target}", tool=tool, args={"target": target}))
            continuations.append(future)
        return continuations


class CachedAPIProvider(TeacherProvider):
    """API provider shell that never hard-codes secrets.

    Full production use should implement `_call_api`. The cache contract is
    already stable, so expensive continuations can be reproduced.
    """

    def __init__(self, model_id: str, api_key_env: str, cache_dir: str | Path) -> None:
        self.model_id = model_id
        self.api_key_env = api_key_env
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def continue_actions(self, prefix: TrajectoryPrefix, horizon: int, k: int) -> List[List[Action]]:
        key = self._cache_key(prefix, horizon, k)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return self._decode(path)
        if not os.environ.get(self.api_key_env):
            raise RuntimeError(
                f"Missing {self.api_key_env}; use MockTeacherProvider or configure API credentials."
            )
        raise NotImplementedError(
            f"API call for {self.model_id} is intentionally adapter-only until credentials and endpoint are configured."
        )

    def _cache_key(self, prefix: TrajectoryPrefix, horizon: int, k: int) -> str:
        payload = json.dumps(
            {
                "model": self.model_id,
                "task_id": prefix.task_id,
                "next_action_index": prefix.next_action_index,
                "horizon": horizon,
                "k": k,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def _decode(self, path: Path) -> List[List[Action]]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [
            [Action(text=item["text"], tool=item["tool"], args=item.get("args", {})) for item in continuation]
            for continuation in raw
        ]

    def write_cache(self, prefix: TrajectoryPrefix, horizon: int, k: int, continuations: Sequence[Sequence[Action]]) -> Path:
        key = self._cache_key(prefix, horizon, k)
        path = self.cache_dir / f"{key}.json"
        payload = [
            [{"text": action.text, "tool": action.tool, "args": dict(action.args)} for action in continuation]
            for continuation in continuations
        ]
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
