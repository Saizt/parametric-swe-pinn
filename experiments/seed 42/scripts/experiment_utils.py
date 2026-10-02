from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set the RNGs used by the current codebase without changing algorithms."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def dump_json(path: str | os.PathLike[str], payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True)


def system_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_default_dtype": str(torch.get_default_dtype()),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    }
    if torch.cuda.is_available():
        meta["cuda_device"] = torch.cuda.get_device_name(0)
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        meta["git_commit"] = None
    return meta


@dataclass
class TimingRecorder:
    """Accumulate named wall-clock timings and persist them as JSON."""

    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.add(name, elapsed)

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + float(seconds)
        self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self) -> dict[str, Any]:
        return {
            name: {
                "seconds": self.totals[name],
                "count": self.counts[name],
                "mean_seconds": self.totals[name] / max(self.counts[name], 1),
            }
            for name in sorted(self.totals)
        }

    def save(self, path: str | os.PathLike[str], extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"timings": self.summary()}
        if extra:
            payload.update(extra)
        dump_json(path, payload)


def snapshot_sources(source_dir: str, destination: str) -> None:
    """Copy the Python/JSON source files that define an experiment."""
    src = Path(source_dir)
    dst = Path(destination)
    dst.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.py", "*.json"):
        for path in src.glob(pattern):
            if path.is_file():
                shutil.copy2(path, dst / path.name)


def save_run_manifest(run_dir: str, args: Any, command: list[str] | None = None) -> None:
    """Write machine-readable resolved configuration and environment metadata."""
    run_dir = str(run_dir)
    payload = vars(args) if hasattr(args, "__dict__") else args
    dump_json(os.path.join(run_dir, "resolved_config.json"), payload)
    dump_json(os.path.join(run_dir, "system_info.json"), system_metadata())
    with open(os.path.join(run_dir, "command.txt"), "w") as f:
        f.write(" ".join(command or sys.argv) + "\n")
