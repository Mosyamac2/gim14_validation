"""
Q7 — Endogenous Relation Drift: Absence of Trade Recovery Mechanism

Test whether trade_intensity recovers after a transient barrier shock
is removed, or permanently settles at a lower level.
"""
from __future__ import annotations
import copy
from ..helpers import (
    gim, load_compact_world, run_steps, make_result, determine_status, Timer,
)


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        world_base = load_compact_world()
        ids = list(world_base.agents.keys())
        if len(ids) < 2:
            return make_result("Q7", "FAIL", {}, notes="Need ≥2 agents")

        a_id, b_id = ids[0], ids[1]

        # ------------------------------------------------------------------
        # Baseline: 10-step run with no shock
        # ------------------------------------------------------------------
        traj_baseline = run_steps(world_base, years=10, seed=42)
        baseline_ti = [
            traj_baseline[t].relations[a_id][b_id].trade_intensity
            for t in range(11)
        ]

        # ------------------------------------------------------------------
        # Shock: inject trade_barrier=0.4 at t=0, remove at t=3
        # We do this by running 3 steps with barrier, then 7 without.
        # ------------------------------------------------------------------
        world_shock = copy.deepcopy(world_base)
        world_shock.relations[a_id][b_id].trade_barrier = 0.4
        world_shock.relations[b_id][a_id].trade_barrier = 0.4

        # Run 3 steps with barrier
        traj_phase1 = run_steps(world_shock, years=3, seed=42)

        # Now remove barrier and run 7 more steps
        world_post = copy.deepcopy(traj_phase1[3])
        world_post.relations[a_id][b_id].trade_barrier = 0.0
        world_post.relations[b_id][a_id].trade_barrier = 0.0

        traj_phase2 = run_steps(world_post, years=7, seed=42)

        # Assemble full shocked trajectory
        shock_ti = []
        for t in range(4):  # t=0..3
            shock_ti.append(traj_phase1[t].relations[a_id][b_id].trade_intensity)
        for t in range(1, 8):  # t=4..10
            shock_ti.append(traj_phase2[t].relations[a_id][b_id].trade_intensity)

        # ------------------------------------------------------------------
        # Assess recovery
        # ------------------------------------------------------------------
        pre_shock_ti = shock_ti[0]
        post_removal_ti = shock_ti[-1]  # at t=10
        trough_ti = min(shock_ti[1:4])  # during barrier period

        if pre_shock_ti > 0:
            recovery_ratio = post_removal_ti / pre_shock_ti
            decay_ratio = post_removal_ti / pre_shock_ti
        else:
            recovery_ratio = 1.0
            decay_ratio = 1.0

        recovers_80pct = recovery_ratio >= 0.80

        conditions = {
            "trade_recovers_80pct": recovers_80pct,
        }

    return make_result(
        test_id="Q7",
        status=determine_status(conditions),
        metrics={
            "pair": f"{a_id} -> {b_id}",
            "pre_shock_trade_intensity": round(pre_shock_ti, 4),
            "trough_trade_intensity": round(trough_ti, 4),
            "post_removal_trade_intensity": round(post_removal_ti, 4),
            "recovery_ratio": round(recovery_ratio, 4),
        },
        flagged_items=[
            f"Trade intensity only recovered to {recovery_ratio*100:.1f}% of pre-shock"
            if not recovers_80pct else "",
        ],
        notes=f"Completed in {t.elapsed:.1f}s",
        details={
            "shock_trajectory": [round(v, 5) for v in shock_ti],
            "baseline_trajectory": [round(v, 5) for v in baseline_ti],
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
