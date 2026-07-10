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

HYPOTHESES: List[Dict[str, object]] = [
    {
        "key": "H1",
        "name": "effectiveness",
        "claim": "LAP improves long-horizon success over same-data same-FLOPs ReAct-SFT and token-MTP.",
        "metrics": ["success_rate", "pass^k", "TGC", "SGC", "resolve_rate", "SR(d)"],
        "experiments": ["main_results", "plandepth_depth_curve"],
        "contributions": ["C1"],
    },
    {
        "key": "H2",
        "name": "mechanism_eld",
        "claim": "ForeAct raises Effective Lookahead Depth under the same Fig.2/Fig.5 probe contract.",
        "metrics": ["effective_lookahead_depth", "future_signal_by_depth"],
        "experiments": ["eld_recovery", "ablation_A", "control_G_predict_past"],
        "contributions": ["C1", "C2"],
    },
    {
        "key": "H3",
        "name": "robustness",
        "claim": "PCR reduces behavioral churn and delayed dead-end entry under stochastic branching.",
        "metrics": ["behavioral_churn_rate", "dead_end_rate", "pass4_minus_pass1"],
        "experiments": ["ablation_B", "ablation_B_prime", "tau2_random_branching"],
        "contributions": ["C2"],
    },
    {
        "key": "H4",
        "name": "efficiency",
        "claim": "Mode A is ReAct-identical at inference; Mode B provides bounded latent rerank overhead.",
        "metrics": ["tokens_per_task", "extra_forward_per_task", "success_overhead_pareto"],
        "experiments": ["mode_A", "mode_B", "ablation_C", "ablation_C_prime"],
        "contributions": ["C3"],
    },
    {
        "key": "H5",
        "name": "boundaries",
        "claim": "ForeAct gains vanish or shrink for shallow tasks, deterministic settings, or bad zeta granularity.",
        "metrics": ["gain_vs_depth", "gain_vs_randomness", "gain_vs_zeta"],
        "experiments": ["plandepth_boundary", "ablation_A_prime", "ablation_D", "ablation_E"],
        "contributions": ["C1", "C2"],
    },
]

CONTRIBUTION_ALIGNMENT: List[Dict[str, object]] = [
    {
        "contribution": "C1",
        "module": "LAP",
        "section": "6.1",
        "ablations": ["A", "A_prime", "token_mtp"],
        "metrics": ["success_rate", "ELD"],
        "paper_guardrail": "action-level vs token-level must remain explicit",
    },
    {
        "contribution": "C2",
        "module": "PCR",
        "section": "6.2",
        "ablations": ["B", "B_prime"],
        "metrics": ["behavioral_churn_rate", "dead_end_rate", "forecast_entropy"],
        "paper_guardrail": "PCR KL direction is KL(sg[q_{t+1}^{h-1}] || q_t^h)",
    },
    {
        "contribution": "C3",
        "module": "Latent Lookahead",
        "section": "6.3",
        "ablations": ["C", "C_prime"],
        "metrics": ["extra_forward_per_task", "success_overhead_pareto"],
        "paper_guardrail": "Mode A remains byte-for-byte ReAct shaped",
    },
]

DESIGN_CHOICES: List[Dict[str, str]] = [
    {
        "choice": "action semantic sketch",
        "rejected_alternative": "token-level or FSP future summary",
        "reason": "decision granularity aligns with feasibility",
        "ablation": "A_prime",
    },
    {
        "choice": "K-rollout soft target",
        "rejected_alternative": "one-hot hard target",
        "reason": "future is a distribution under stochastic branching",
        "ablation": "E",
    },
    {
        "choice": "forward KL stop-gradient PCR",
        "rejected_alternative": "reverse KL or no stop-gradient",
        "reason": "only the forward direction recovers the branch marginal in expectation",
        "ablation": "B",
    },
    {
        "choice": "branch-aware weighting",
        "rejected_alternative": "uniform weighting",
        "reason": "high-entropy futures are partly unknowable",
        "ablation": "B_prime",
    },
    {
        "choice": "train-time-only heads",
        "rejected_alternative": "mandatory inference heads",
        "reason": "default Mode A must have zero inference overhead",
        "ablation": "C",
    },
]

REMARK_1_CORNERS: List[Dict[str, str]] = [
    {"method": "greedy ReAct", "degenerate_corner": "H=0"},
    {"method": "token-MTP", "degenerate_corner": "zeta=identity, depth measured in tokens"},
    {"method": "FSP future summary", "degenerate_corner": "bag-of-future sketches, no per-depth heads"},
    {"method": "VideoPlan-style action MTP", "degenerate_corner": "K=1, deterministic one-hot targets, mu=0"},
]


def registry() -> Dict[str, list]:
    return {
        "baselines": [spec.__dict__ for spec in BASELINES],
        "ablations": [spec.__dict__ for spec in ABLATIONS],
        "hypotheses": HYPOTHESES,
        "contribution_alignment": CONTRIBUTION_ALIGNMENT,
        "design_choices": DESIGN_CHOICES,
        "remark_1_degenerate_corners": REMARK_1_CORNERS,
    }
