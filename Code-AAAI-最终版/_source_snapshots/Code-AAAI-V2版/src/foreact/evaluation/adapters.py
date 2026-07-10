"""External benchmark harness adapters with graceful skips."""

from __future__ import annotations

import importlib.util
import importlib
import subprocess
from pathlib import Path
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
    fallback_paths: tuple[str, ...] = ()
    fallback_interpreters: tuple[str, ...] = ()

    def status(self) -> HarnessStatus:
        import_error = ""
        if self.package_name and importlib.util.find_spec(self.package_name):
            try:
                importlib.import_module(self.package_name)
            except Exception as exc:
                import_error = f"installed but import failed: {exc}"
            else:
                return HarnessStatus(self.package_name, True, "available")
        for interpreter in self.fallback_interpreters:
            path = Path(interpreter)
            if path.exists():
                completed = subprocess.run(
                    [str(path), "-c", f"import {self.package_name}; print('ok')"],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if completed.returncode == 0:
                    return HarnessStatus(
                        self.package_name or self.__class__.__name__,
                        True,
                        f"available via isolated interpreter {interpreter}",
                    )
                import_error = f"isolated interpreter {interpreter} import failed: {completed.stderr.strip()[-500:]}"
        for path in self.fallback_paths:
            if Path(path).exists():
                return HarnessStatus(
                    self.package_name or self.__class__.__name__,
                    True,
                    f"source present at {path}; install editable package before official runs",
                )
        message = import_error or self.setup_hint
        return HarnessStatus(self.package_name or self.__class__.__name__, False, message)


class Tau2BenchAdapter(BenchmarkAdapter):
    package_name = "tau2"
    setup_hint = "Install tau2-bench and configure airline/retail task data; smoke runs skip this harness."
    fallback_paths = ("third_party/benchmarks/tau2-bench/src/tau2",)


class AppWorldAdapter(BenchmarkAdapter):
    package_name = "appworld"
    setup_hint = "Install AppWorld and download official train/test splits."
    fallback_interpreters = (".venv-appworld/bin/python",)


class SweGymAdapter(BenchmarkAdapter):
    package_name = ""
    setup_hint = "Fetch SWE-Gym repository/data; it is script/data based rather than an importable package."
    fallback_paths = ("third_party/benchmarks/SWE-Gym/README.md", "datasets/swe-gym/DATASET_READY.json")


class SweBenchVerifiedAdapter(BenchmarkAdapter):
    package_name = "swebench"
    setup_hint = "Install SWE-bench and configure the Verified evaluation harness."
    fallback_paths = ("third_party/benchmarks/SWE-bench/swebench",)


def external_harness_status() -> Dict[str, dict]:
    adapters = {
        "tau2_bench": Tau2BenchAdapter(),
        "appworld": AppWorldAdapter(),
        "swe_gym": SweGymAdapter(),
        "swe_bench_verified": SweBenchVerifiedAdapter(),
    }
    return {name: adapter.status().__dict__ for name, adapter in adapters.items()}
