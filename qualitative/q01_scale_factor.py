"""
Q1 — Production Function Energy Exponent and Scale-Factor Lock-In

Inspect the distribution of _scale_factor across agents and measure
how initial-state perturbations propagate through the frozen factor.
"""
from __future__ import annotations
import copy
import statistics
from ..helpers import (
    gim, load_operational_world, run_steps,
    perturb_agent_field, agent_gdp, make_result, determine_status, Timer,
)


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        world = load_operational_world()
        g = gim()

        # ------------------------------------------------------------------
        # Step 1: trigger _scale_factor initialisation by running 1 step
        # ------------------------------------------------------------------
        traj_init = run_steps(world, years=1, seed=42)
        world_after_1 = traj_init[1]

        # Collect scale factors
        scale_factors: dict[str, float] = {}
        for aid, agent in world_after_1.agents.items():
            sf = getattr(agent.economy, "_scale_factor", None)
            scale_factors[aid] = sf if sf is not None else float("nan")

        valid_sfs = [v for v in scale_factors.values() if v == v]  # exclude NaN
        sf_mean = statistics.mean(valid_sfs) if valid_sfs else 0.0
        sf_std = statistics.pstdev(valid_sfs) if len(valid_sfs) > 1 else 0.0
        sf_cv = sf_std / sf_mean if sf_mean > 0 else float("inf")
        extreme_high = {k: v for k, v in scale_factors.items() if v > 2.0}
        extreme_low = {k: v for k, v in scale_factors.items() if v < 0.3 and v == v}

        # ------------------------------------------------------------------
        # Step 2: GDP propagation from ±20% capital perturbation
        # ------------------------------------------------------------------
        agents_sorted = sorted(
            world.agents.values(),
            key=lambda a: abs(scale_factors.get(a.id, 1.0) - sf_mean),
            reverse=True,
        )
        test_agents = [a.id for a in agents_sorted[:5]]

        perturbation_results = []
        for aid in test_agents:
            baseline_traj = run_steps(world, years=5, seed=42)
            baseline_gdp_t5 = agent_gdp(baseline_traj[5], aid)

            for direction, factor in [("up", 1.20), ("down", 0.80)]:
                w_pert = perturb_agent_field(world, aid, "economy.capital", factor)
                pert_traj = run_steps(w_pert, years=5, seed=42)
                pert_gdp_t5 = agent_gdp(pert_traj[5], aid)
                divergence = abs(pert_gdp_t5 - baseline_gdp_t5) / max(baseline_gdp_t5, 1e-12)
                perturbation_results.append({
                    "agent_id": aid,
                    "agent_name": world.agents[aid].name,
                    "direction": direction,
                    "scale_factor": scale_factors.get(aid),
                    "baseline_gdp_t5": round(baseline_gdp_t5, 4),
                    "perturbed_gdp_t5": round(pert_gdp_t5, 4),
                    "gdp_divergence_pct": round(divergence * 100, 2),
                })

        # ------------------------------------------------------------------
        # Assess pass / fail
        # ------------------------------------------------------------------
        cv_ok = sf_cv < 1.0
        bounded = sum(
            1 for r in perturbation_results if r["gdp_divergence_pct"] < 15.0
        )
        bounded_ok = bounded >= 0.8 * len(perturbation_results)

        conditions = {
            "scale_factor_cv_lt_1": cv_ok,
            "perturbation_bounded_80pct": bounded_ok,
        }

    return make_result(
        test_id="Q1",
        status=determine_status(conditions),
        metrics={
            "scale_factor_mean": round(sf_mean, 4),
            "scale_factor_std": round(sf_std, 4),
            "scale_factor_cv": round(sf_cv, 4),
            "extreme_high_count": len(extreme_high),
            "extreme_low_count": len(extreme_low),
            "perturbation_bounded_count": bounded,
            "perturbation_total_count": len(perturbation_results),
        },
        flagged_items=[
            f"extreme_high: {k} = {round(v, 3)}" for k, v in extreme_high.items()
        ] + [
            f"extreme_low: {k} = {round(v, 3)}" for k, v in extreme_low.items()
        ],
        notes=f"Completed in {t.elapsed:.1f}s",
        details={
            "perturbation_results": perturbation_results,
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
