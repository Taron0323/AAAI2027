"""External benchmark harness adapters with graceful skips."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class HarnessStatus:
    name: str
    available: bool
    message: str


class BenchmarkAdapter:
    package_name: str = ""
    setup_hint: str = ""

    def status(self) -> HarnessStatus:
        available = bool(self.package_name and importlib.util.find_spec(self.package_name))
        if available:
            return HarnessStatus(self.package_name, True, "available")
        return HarnessStatus(self.package_name or self.__class__.__name__, False, self.setup_hint)


class Tau2BenchAdapter(BenchmarkAdapter):
    package_name = "tau2"
    setup_hint = "Install tau2-bench and configure airline/retail task data; smoke runs skip this harness."


class AppWorldAdapter(BenchmarkAdapter):
    package_name = "appworld"
    setup_hint = "Install AppWorld and download official train/test splits."


class SweGymAdapter(BenchmarkAdapter):
    package_name = "swegym"
    setup_hint = "Install SWE-Gym for non-overlapping software-agent training trajectories."


class SweBenchVerifiedAdapter(BenchmarkAdapter):
    package_name = "swebench"
    setup_hint = "Install SWE-bench and configure the Verified evaluation harness."


def external_harness_status() -> Dict[str, dict]:
    adapters = {
        "tau2_bench": Tau2BenchAdapter(),
        "appworld": AppWorldAdapter(),
        "swe_gym": SweGymAdapter(),
        "swe_bench_verified": SweBenchVerifiedAdapter(),
    }
    return {name: adapter.status().__dict__ for name, adapter in adapters.items()}

