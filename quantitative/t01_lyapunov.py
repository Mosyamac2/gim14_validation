"""
T1 — GDP Trajectory Stability Under Initial-Condition Perturbation (Lyapunov-Type Test)

Measure sensitivity of the 5-year GDP trajectory to ±1% perturbations
of 6 initial-state variables for 10 representative agents.
"""
from __future__ import annotations
import copy
from ..helpers import (
    gim, load_operational_world, run_steps,
    perturb_agent_field, agent_gdp,
    make_result, determine_status, Timer,
)


PERTURB_FIELDS = [
    "economy.gdp",
    "economy.capital",
    "economy.population",
    "society.trust_gov",
    "society.social_tension",
    "economy.public_debt",
]
PERTURBATION = 0.01  # ±1%


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        world = load_operational_world()

        # Select 10 agents: top-5 and bottom-5 by GDP
        sorted_agents = sorted(
            world.agents.values(),
            key=lambda a: a.economy.gdp,
            reverse=True,
        )
        test_agents = sorted_agents[:5] + sorted_agents[-5:]
        test_ids = [a.id for a in test_agents]

        # Baseline run
        traj_base = run_steps(world, years=5, seed=42)

        results = []
        flagged = []

        for aid in test_ids:
            baseline_gdp_t5 = agent_gdp(traj_base[5], aid)
            agent_name = world.agents[aid].name
            debt_gdp = world.agents[aid].economy.public_debt / max(world.agents[aid].economy.gdp, 1e-9)
            near_crisis = debt_gdp > 1.0 or world.agents[aid].society.trust_gov < 0.25
            threshold = 15.0 if near_crisis else 5.0

            for field in PERTURB_FIELDS:
                for direction, factor in [("up", 1.0 + PERTURBATION), ("down", 1.0 - PERTURBATION)]:
                    w_pert = perturb_agent_field(world, aid, field, factor)
                    traj_pert = run_steps(w_pert, years=5, seed=42)
                    pert_gdp_t5 = agent_gdp(traj_pert[5], aid)

                    divergence = abs(pert_gdp_t5 - baseline_gdp_t5) / max(baseline_gdp_t5, 1e-12)
                    amp_ratio = divergence / PERTURBATION

                    row = {
                        "agent_id": aid,
                        "agent_name": agent_name,
                        "field": field,
                        "direction": direction,
                        "gdp_divergence_t5_pct": round(divergence * 100, 3),
                        "amp_ratio": round(amp_ratio, 2),
                        "near_crisis": near_crisis,
                        "threshold": threshold,
                    }
                    results.append(row)

                    if amp_ratio > 20:
                        flagged.append(
                            f"CRITICAL: {agent_name}/{field}/{direction} AmpRatio={amp_ratio:.1f}"
                        )
                    elif amp_ratio > threshold:
                        flagged.append(
                            f"WARN: {agent_name}/{field}/{direction} AmpRatio={amp_ratio:.1f} > {threshold}"
                        )

        # Assess
        non_crisis = [r for r in results if not r["near_crisis"]]
        crisis = [r for r in results if r["near_crisis"]]

        non_crisis_ok = all(r["amp_ratio"] < 5 for r in non_crisis) if non_crisis else True
        crisis_ok = all(r["amp_ratio"] < 15 for r in crisis) if crisis else True
        no_extreme = all(r["amp_ratio"] < 20 for r in results)

        max_amp = max((r["amp_ratio"] for r in results), default=0)
        mean_amp = sum(r["amp_ratio"] for r in results) / max(len(results), 1)

        conditions = {
            "non_crisis_amp_lt_5": non_crisis_ok,
            "crisis_amp_lt_15": crisis_ok,
            "no_extreme_gt_20": no_extreme,
        }

    return make_result(
        test_id="T1",
        status=determine_status(conditions),
        metrics={
            "total_perturbations": len(results),
            "max_amp_ratio": round(max_amp, 2),
            "mean_amp_ratio": round(mean_amp, 2),
            "non_crisis_agents": len(set(r["agent_id"] for r in non_crisis)),
            "crisis_agents": len(set(r["agent_id"] for r in crisis)),
            "flagged_count": len(flagged),
        },
        flagged_items=flagged[:20],  # cap at 20
        notes=f"Completed in {t.elapsed:.1f}s. {len(results)} perturbation runs.",
        details={
            "conditions": {k: bool(v) for k, v in conditions.items()},
            "sample_results": results[:30],  # first 30 for brevity
        },
    )
