"""
Q5 — Debt Crisis Onset Haircut Paradox

Verify that the debt-crisis onset haircut (debt *= 0.60, GDP *= 0.90)
mechanically resolves the crisis condition within 1-2 steps for
realistic initial debt/GDP ratios.
"""
from __future__ import annotations
import copy
from ..helpers import (
    gim, load_compact_world, run_steps, make_result, determine_status, Timer,
)


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        world_base = load_compact_world()
        cal = gim()["cal"]

        # Pick an agent to stress
        test_agent_id = list(world_base.agents.keys())[0]

        initial_ratios = [1.25, 1.50, 1.80, 2.00, 2.50, 3.00]
        results = []

        for ratio in initial_ratios:
            world = copy.deepcopy(world_base)
            agent = world.agents[test_agent_id]

            # Set up crisis conditions
            agent.economy.public_debt = agent.economy.gdp * ratio
            agent.risk.debt_crisis_prone = 0.9
            agent.risk.regime_stability = 0.4
            # Need interest_rate > 0.12 for trigger
            # Increasing debt/gdp and debt_crisis_prone should push rate up

            crisis_years_log = []
            traj = run_steps(world, years=6, seed=42, enable_extreme_events=False)

            for step in range(7):
                a = traj[step].agents[test_agent_id]
                debt_gdp = a.economy.public_debt / max(a.economy.gdp, 1e-9)
                active_years = a.risk.debt_crisis_active_years
                crisis_years_log.append({
                    "t": step,
                    "debt_gdp": round(debt_gdp, 4),
                    "active_years": active_years,
                })

            max_active = max(e["active_years"] for e in crisis_years_log)
            persists_3y = max_active >= 3

            results.append({
                "initial_debt_gdp": ratio,
                "max_crisis_active_years": max_active,
                "persists_3y": persists_3y,
                "trajectory": crisis_years_log,
            })

        # Find minimum ratio for 3-year persistence
        min_ratio_3y = None
        for r in results:
            if r["persists_3y"]:
                min_ratio_3y = r["initial_debt_gdp"]
                break

        # Check: crisis should persist ≥ 3 years at debt/GDP 1.2–1.5
        near_threshold_persists = any(
            r["persists_3y"] for r in results
            if r["initial_debt_gdp"] <= 1.50
        )

        conditions = {
            "crisis_persists_near_threshold": near_threshold_persists,
        }

    return make_result(
        test_id="Q5",
        status=determine_status(conditions),
        metrics={
            "min_ratio_for_3y_persistence": min_ratio_3y,
            "DEBT_CRISIS_DEBT_MULT": cal.DEBT_CRISIS_DEBT_MULT,
            "DEBT_CRISIS_GDP_MULT": cal.DEBT_CRISIS_GDP_MULT,
            "DEBT_CRISIS_EXIT_THRESHOLD": cal.DEBT_CRISIS_EXIT_THRESHOLD,
            "onset_ratio_after_haircut_at_1.25": round(
                1.25 * cal.DEBT_CRISIS_DEBT_MULT / cal.DEBT_CRISIS_GDP_MULT, 4
            ),
        },
        flagged_items=[
            "Debt crisis clears in ≤2 steps at debt/GDP=1.25 due to onset haircut"
            if not near_threshold_persists else "",
        ],
        notes=f"Completed in {t.elapsed:.1f}s",
        details={
            "results_by_ratio": results,
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
