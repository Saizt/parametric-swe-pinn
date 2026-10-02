from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any


def _regimes(values):
    return [(float(v[0]), float(v[1]), str(v[2]) if len(v) > 2 else "perpendicular") for v in values]


def run_evaluation_plan(pinn, plan: dict[str, Any]) -> dict[str, Any]:
    """Run a JSON-compatible evaluation plan and return produced artifacts."""
    artifacts: dict[str, Any] = {}

    cases = plan.get("cases")
    times = plan.get("times")
    if cases and times:
        results = []
        for t in [float(x) for x in times]:
            for regime in _regimes(cases):
                results.append(pinn.evaluate(test_t=t, test_h=regime, plot=bool(plan.get("individual_plots", False))))
        path = Path(pinn.checkpoint_folder) / plan.get("results_file", "test_results.json")
        with path.open("w") as f:
            json.dump(results, f, indent=2)
        artifacts["test_results"] = str(path)

    paper = plan.get("paper_metrics")
    if paper and paper.get("enabled", True):
        kwargs = {k: v for k, v in paper.items() if k != "enabled"}
        artifacts["paper_metrics"] = pinn.paper_style_eval(**kwargs)

    slice_eval = plan.get("slice_evaluation")
    if slice_eval and slice_eval.get("enabled", True):
        hL = float(slice_eval["hL"])
        hR_values = np.linspace(
            float(slice_eval["hR_min"]),
            float(slice_eval["hR_max"]),
            int(slice_eval["n_hR"]),
        )

        artifacts["slice_evaluation"] = pinn.paper_style_eval(
            sampling="grid",
            grid_hL=[hL],
            grid_hR=[float(x) for x in hR_values],
            include_H1=bool(slice_eval.get("include_H1", False)),
            Nx=int(slice_eval.get("Nx", 402)),
            t_final=float(slice_eval.get("t_final", 2.5)),
            dt=float(slice_eval.get("dt", 1e-4)),
            sample_interval=float(slice_eval.get("sample_interval", 0.01)),
            fields=slice_eval.get("fields", ["h", "q", "state"]),
            aggregate=slice_eval.get("aggregate", "mean"),
            dam_shape=slice_eval.get("dam_shape", "perpendicular"),
            save_name=slice_eval.get("save_name", "slice_25"),
        )

    for i, spec in enumerate(plan.get("profile_plots", [])):
        kwargs = dict(spec)
        kwargs["test_h"] = _regimes(kwargs["test_h"])
        artifacts[f"profile_plots_{i}"] = pinn.plot_trom_style_final_profiles(**kwargs)

    for i, spec in enumerate(plan.get("error_evolution_plots", [])):
        kwargs = dict(spec)
        kwargs["test_h"] = _regimes(kwargs["test_h"])
        if kwargs.get("extra_table_h") is not None:
            kwargs["extra_table_h"] = _regimes(kwargs["extra_table_h"])
        artifacts[f"error_evolution_{i}"] = pinn.plot_trom_style_error_evolution(**kwargs)

    return artifacts
