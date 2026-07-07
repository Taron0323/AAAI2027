"""Named baselines and ablations required by the ForeAct paper plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    name: str
    role: str
    implemented: str


BASELINES: List[BaselineSpec] = [
    BaselineSpec("react", "Qwen3-Instruct + ReAct prompt", "vanilla", "adapter"),
    BaselineSpec("react_sft", "ReAct-SFT", "same teacher data/FLOPs", "adapter"),
    BaselineSpec("plan_and_act", "Plan-and-Act / Pre-Act", "explicit text planning", "smoke"),
    BaselineSpec("lats_webdreamer", "LATS / WebDreamer-style search", "explicit lookahead", "smoke"),
    BaselineSpec("token_mtp", "AR + token-level MTP", "critical action-vs-token baseline", "adapter"),
    BaselineSpec("gpt55_reference", "GPT-5.5 zero-shot ReAct", "frontier reference only", "external"),
]

ABLATIONS: List[BaselineSpec] = [
    BaselineSpec("A", "remove forecast heads", "contribution 1", "config"),
    BaselineSpec("A_prime_token", "zeta=token", "granularity spectrum", "config"),
    BaselineSpec("A_prime_fsp", "FSP-style future summary", "nearest-neighbor control", "config"),
    BaselineSpec("A_prime_type", "operation type only", "granularity spectrum", "config"),
    BaselineSpec("A_prime_type_arg", "operation type x primary argument", "main setting", "config"),
    BaselineSpec("A_prime_vq512", "VQ-512 learned sketches", "manual-schema-free variant", "stub"),
    BaselineSpec("B", "remove PCR", "contribution 2", "config"),
    BaselineSpec("B_prime", "remove branch-aware weighting", "PCR component", "config"),
    BaselineSpec("C", "disable latent rerank", "system mode A", "config"),
    BaselineSpec("C_prime", "success head vs forecast head scoring", "rerank component", "config"),
    BaselineSpec("D", "H sweep", "horizon", "config"),
    BaselineSpec("E", "K sweep", "soft target necessity", "config"),
    BaselineSpec("E_prime", "GPT-5.5 second teacher subset", "teacher independence", "external"),
    BaselineSpec("F", "curriculum/lambda/mu sweep", "training stability", "config"),
    BaselineSpec("G_predict_past", "predict past sketches", "anti-confounding control", "config"),
    BaselineSpec("G_flops_matched", "FLOPs-matched SFT", "anti-confounding control", "config"),
]


def registry() -> Dict[str, list]:
    return {
        "baselines": [spec.__dict__ for spec in BASELINES],
        "ablations": [spec.__dict__ for spec in ABLATIONS],
    }
