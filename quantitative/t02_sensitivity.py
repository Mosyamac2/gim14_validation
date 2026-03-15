"""
T2 — Parameter Sensitivity Analysis: Prior-Heavy Calibration Parameters

Sweep 8 key [PRIOR]-tagged parameters and measure their marginal effect
on 5 output metrics to identify calibration priorities.
"""
from __future__ import annotations
import copy
import numpy as np
from ..helpers import (
    gim, load_compact_world, run_steps, patch_param,
    global_gdp, global_avg_trust, global_avg_tension, global_co2,
    count_distressed_agents,
    make_result, determine_status, Timer,
)


PARAM_SWEEP = [
    ("GINI_FISCAL_SENS",                  -60.0,  -90.0,  -30.0),
    ("CRISK_TEMP_SENSITIVITY",             0.45,   0.20,   0.70),
    ("STRUCTURAL_TRANSITION_POLICY_SENS",  0.50,   0.25,   0.75),
    ("DAMAGE_QUAD_COEFF",                  0.006,  0.003,  0.012),
    ("TRUST_TENSION_SENS",                -0.08,  -0.15,  -0.04),
    ("DEBT_SPREAD_QUADRATIC",              0.10,   0.05,   0.20),
    ("EVENT_MAX_EXTRA_PROB",               0.07,   0.03,   0.14),
    ("SAVINGS_STABILITY_SENS",             0.60,   0.30,   0.90),
]

N_POINTS = 5
METRIC_NAMES = ["global_gdp", "global_trust", "global_tension", "global_co2", "distressed_count"]


def _extract_metrics(world) -> dict[str, float]:
    return {
        "global_gdp": global_gdp(world),
        "global_trust": global_avg_trust(world),
        "global_tension": global_avg_tension(world),
        "global_co2": global_co2(world),
        "distressed_count": float(count_distressed_agents(world)),
    }


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        world = load_compact_world()

        # Baseline run
        traj_base = run_steps(world, years=5, seed=42)
        base_metrics = _extract_metrics(traj_base[5])

        sensitivity_matrix = {}
        all_runs = []

        for param_name, baseline, lo, hi in PARAM_SWEEP:
            sweep_values = [lo + i * (hi - lo) / (N_POINTS - 1) for i in range(N_POINTS)]
            param_metrics = []

            for val in sweep_values:
                with patch_param(param_name, val):
                    traj = run_steps(world, years=5, seed=42)
                metrics = _extract_metrics(traj[5])
                param_metrics.append({"value": val, "metrics": metrics})

            # Compute sensitivity index S = [M(max) - M(min)] / M(baseline)
            sens = {}
            for m in METRIC_NAMES:
                m_values = [pm["metrics"][m] for pm in param_metrics]
                m_range = max(m_values) - min(m_values)
                m_base = base_metrics[m]
                if abs(m_base) > 1e-9:
                    sens[m] = round(m_range / abs(m_base), 4)
                else:
                    sens[m] = round(m_range, 4)

            # Elasticity at baseline
            elasticity = {}
            # Find the two points closest to baseline
            closest_idx = min(range(N_POINTS), key=lambda i: abs(sweep_values[i] - baseline))
            if closest_idx < N_POINTS - 1:
                dm = {m: param_metrics[closest_idx + 1]["metrics"][m] - param_metrics[closest_idx]["metrics"][m]
                      for m in METRIC_NAMES}
                dtheta = sweep_values[closest_idx + 1] - sweep_values[closest_idx]
                for m in METRIC_NAMES:
                    m_base = base_metrics[m]
                    if abs(m_base) > 1e-9 and abs(dtheta) > 1e-12 and abs(baseline) > 1e-12:
                        elasticity[m] = round((dm[m] / m_base) / (dtheta / baseline), 4)
                    else:
                        elasticity[m] = 0.0

            sensitivity_matrix[param_name] = {
                "sensitivity_index": sens,
                "elasticity": elasticity,
                "max_sensitivity": max(abs(v) for v in sens.values()),
            }

            all_runs.append({
                "param": param_name,
                "baseline": baseline,
                "range": [lo, hi],
                "sweep_results": [
                    {"value": round(pm["value"], 6),
                     **{m: round(pm["metrics"][m], 4) for m in METRIC_NAMES}}
                    for pm in param_metrics
                ],
            })

        # Rank by max sensitivity
        ranked = sorted(
            sensitivity_matrix.items(),
            key=lambda item: item[1]["max_sensitivity"],
            reverse=True,
        )

        # Check thresholds
        any_fragile = any(
            any(abs(v) > 0.30 for v in entry["sensitivity_index"].values()
                if isinstance(v, (int, float)))
            for entry in sensitivity_matrix.values()
        )
        responsive = sum(
            1 for entry in sensitivity_matrix.values()
            if entry["max_sensitivity"] > 0.05
        )

        conditions = {
            "no_fragile_gt_030": not any_fragile,
            "responsive_ge_3": responsive >= 3,
        }

    return make_result(
        test_id="T2",
        status=determine_status(conditions),
        metrics={
            "top_priority_param": ranked[0][0] if ranked else None,
            "top_priority_max_S": round(ranked[0][1]["max_sensitivity"], 4) if ranked else None,
            "responsive_param_count": responsive,
            "fragile_param_exists": any_fragile,
        },
        flagged_items=[
            f"{name}: max|S|={entry['max_sensitivity']:.3f}"
            for name, entry in ranked[:3]
        ],
        notes=f"Completed in {t.elapsed:.1f}s. {len(PARAM_SWEEP)} params × {N_POINTS} points = {len(PARAM_SWEEP) * N_POINTS} runs.",
        details={
            "sensitivity_matrix": sensitivity_matrix,
            "ranking": [(name, round(entry["max_sensitivity"], 4)) for name, entry in ranked],
            "base_metrics": {k: round(v, 4) for k, v in base_metrics.items()},
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
