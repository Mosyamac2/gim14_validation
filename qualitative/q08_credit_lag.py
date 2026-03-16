"""
Q8 — Credit Rating → Interest Rate Feedback Lag

Verify whether the credit-zone premium affects interest rates with
a 0-year or 1-year lag due to step ordering.
"""
from __future__ import annotations
import copy
from ..helpers import (
    gim, get_cal, load_compact_world, run_steps, make_result, determine_status, Timer,
)


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        world_base = load_compact_world()
        cal = get_cal()
        from gim.core.economy import compute_effective_interest_rate

        # Pick an agent and stress it
        test_id = list(world_base.agents.keys())[0]
        world = copy.deepcopy(world_base)
        agent = world.agents[test_id]
        agent.economy.public_debt = agent.economy.gdp * 1.0
        agent.risk.debt_crisis_prone = 0.85
        agent.risk.regime_stability = 0.35

        # Record the initial credit zone
        initial_zone = agent.credit_zone

        # Run 5 steps, tracking rate and zone at each step
        traj = run_steps(world, years=5, seed=42)

        step_records = []
        for i in range(6):
            a = traj[i].agents[test_id]
            rate = compute_effective_interest_rate(a, traj[i])
            debt_gdp = a.economy.public_debt / max(a.economy.gdp, 1e-9)
            step_records.append({
                "t": i,
                "credit_zone": a.credit_zone,
                "credit_rating": a.credit_rating,
                "effective_rate": round(rate, 5),
                "debt_gdp": round(debt_gdp, 4),
            })

        # Check for lag: find first step where zone worsens
        zone_order = {"prime": 0, "investment": 1, "sub_investment": 2,
                      "distressed": 3, "default": 4}
        first_worsen_step = None
        first_rate_increase_step = None

        for i in range(1, len(step_records)):
            prev_zone = zone_order.get(step_records[i-1]["credit_zone"], 1)
            curr_zone = zone_order.get(step_records[i]["credit_zone"], 1)
            if curr_zone > prev_zone and first_worsen_step is None:
                first_worsen_step = i

            if (step_records[i]["effective_rate"] > step_records[i-1]["effective_rate"] + 0.005
                    and first_rate_increase_step is None):
                first_rate_increase_step = i

        # Lag = rate increase step - zone worsen step
        if first_worsen_step is not None and first_rate_increase_step is not None:
            lag = first_rate_increase_step - first_worsen_step
        else:
            lag = None

        # Compare lagged vs non-lagged debt trajectory (conceptual)
        # In non-lagged, credit update would run before public_finances
        lag_documented = lag is not None and lag <= 1

        conditions = {
            "lag_is_documented_or_zero": lag_documented or lag is None,
        }

    return make_result(
        test_id="Q8",
        status="WARN" if lag is not None and lag >= 1 else "PASS",
        metrics={
            "initial_credit_zone": initial_zone,
            "first_zone_worsen_step": first_worsen_step,
            "first_rate_increase_step": first_rate_increase_step,
            "estimated_lag_steps": lag,
        },
        flagged_items=[
            f"Credit zone → interest rate lag = {lag} step(s)"
            if lag is not None and lag >= 1 else "",
        ],
        notes=f"Completed in {t.elapsed:.1f}s. Step ordering: credit_rating at step 29, public_finances at step 21.",
        details={
            "step_records": step_records,
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
