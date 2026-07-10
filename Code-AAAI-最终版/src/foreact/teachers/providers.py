"""Teacher provider abstraction with mock and API-ready implementations."""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from foreact.types import Action, TrajectoryPrefix, action_history_text


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
    """OpenAI-compatible chat-completions provider with stable JSON cache."""

    def __init__(
        self,
        model_id: str,
        api_key_env: str,
        cache_dir: str | Path,
        base_url: str | None = None,
        timeout_s: int = 120,
        max_retries: int = 3,
    ) -> None:
        self.model_id = model_id
        self.api_key_env = api_key_env
        self.cache_dir = Path(cache_dir)
        self.base_url = base_url or _default_base_url(model_id)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
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
        continuations = self._call_api(prefix, horizon, k)
        self.write_cache(prefix, horizon, k, continuations)
        return continuations

    def _cache_key(self, prefix: TrajectoryPrefix, horizon: int, k: int) -> str:
        payload = json.dumps(
            {
                "model": self.model_id,
                "task_id": prefix.task_id,
                "next_action_index": prefix.next_action_index,
                "horizon": horizon,
                "k": k,
                "history": action_history_text(prefix.goal, prefix.actions, prefix.observations),
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

    def _call_api(self, prefix: TrajectoryPrefix, horizon: int, k: int) -> List[List[Action]]:
        prompt = _teacher_prompt(prefix, horizon, k)
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate future tool/action continuations for ForeAct training. "
                        "Return only valid JSON; do not include markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.8,
            "n": 1,
            "response_format": {"type": "json_object"},
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._post_json(payload)
                content = _extract_chat_content(response)
                return _parse_teacher_json(content, horizon=horizon, k=k)
            except (ValueError, KeyError, HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Teacher API failed for {self.model_id}: {last_error}") from last_error

    def _post_json(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url,
            data=data,
            headers={
                "Authorization": f"Bearer {os.environ[self.api_key_env]}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))


def _default_base_url(model_id: str) -> str:
    if model_id.startswith("deepseek"):
        return os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")


def _teacher_prompt(prefix: TrajectoryPrefix, horizon: int, k: int) -> str:
    history = action_history_text(prefix.goal, prefix.actions, prefix.observations)
    return (
        "Given the ReAct history below, sample plausible future actions only.\n"
        f"Generate exactly {k} continuations, each with at most {horizon} actions.\n"
        "Each action must have fields: text, tool, args.\n"
        "Return JSON with shape {\"continuations\": [[{\"text\": ..., \"tool\": ..., \"args\": {...}}]]}.\n\n"
        f"{history}"
    )


def _extract_chat_content(response: Mapping[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    if not isinstance(message, Mapping) or "content" not in message:
        raise ValueError("chat response has no message content")
    return str(message["content"])


def _parse_teacher_json(content: str, horizon: int, k: int) -> List[List[Action]]:
    data = json.loads(content)
    raw = data.get("continuations")
    if not isinstance(raw, list):
        raise ValueError("teacher JSON missing continuations list")
    continuations: List[List[Action]] = []
    for continuation in raw[:k]:
        if not isinstance(continuation, list):
            continue
        actions = []
        for item in continuation[:horizon]:
            if not isinstance(item, Mapping):
                continue
            text = str(item.get("text", ""))
            tool = str(item.get("tool") or (text.split()[0] if text.split() else "unknown"))
            args = item.get("args", {})
            if not isinstance(args, Mapping):
                args = {"value": str(args)}
            actions.append(Action(text=text or tool, tool=tool, args={str(key): str(value) for key, value in args.items()}))
        continuations.append(actions)
    if len(continuations) < k:
        raise ValueError(f"teacher returned {len(continuations)} continuations, expected {k}")
    return continuations
