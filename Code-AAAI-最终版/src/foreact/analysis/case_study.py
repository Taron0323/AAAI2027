"""Case-study artifact construction."""

from __future__ import annotations

from typing import List

from foreact.environments.plandepth import PlanDepthTask
from foreact.training.toy_trainer import ToyForeActModel
from foreact.types import Action
from foreact.zeta import SketchMapper
from foreact.data.soft_targets import deterministic_features


def plandepth_case_rows(
    task: PlanDepthTask,
    actions: List[Action],
    model: ToyForeActModel,
    mapper: SketchMapper,
) -> List[dict]:
    rows: List[dict] = []
    history: List[Action] = []
    sketches = list(mapper.sketches)
    dead_end_indices = [
        idx
        for idx, sketch in enumerate(sketches)
        if sketch.startswith("commit") or sketch == "<unk>"
    ]
    for step, action in enumerate(actions):
        text = task.goal + "\n" + "\n".join(item.text for item in history)
        probs = model.forecast_probs(deterministic_features(text, model.config.hidden_dim))
        forecast = []
        for depth in range(model.config.horizon):
            dead_mass = float(sum(probs[depth, idx] for idx in dead_end_indices))
            top_idx = int(probs[depth].argmax())
            forecast.append(
                {
                    "depth": depth + 1,
                    "top_sketch": sketches[top_idx],
                    "top_probability": float(probs[depth, top_idx]),
                    "dead_end_mass": dead_mass,
                }
            )
        rows.append(
            {
                "step": step + 1,
                "action": action.text,
                "sketch": mapper.encode(action),
                "dead_end": action.args.get("trap") == "true",
                "forecast": forecast,
            }
        )
        history.append(action)
    return rows
