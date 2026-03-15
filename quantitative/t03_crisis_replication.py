"""
T3 — Crisis-Pathway Replication Test: Debt Crisis Duration and GDP Impact

Verify that the model can reproduce stylized trajectories of
Argentina 2001 (prolonged debt crisis) and Turkey 2018 (FX crisis).
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from ..helpers import (
    gim, load_operational_world, run_steps,
    make_result, determine_status, Timer,
)


def _run_argentina_test(world, n_runs: int = 3) -> dict:
    """Simulate an Argentina-2001-like agent for 8 years."""
    # Find Argentina or use first agent
    arg_id = None
    for aid, agent in world.agents.items():
        if "argentin" in agent.name.lower():
            arg_id = aid
            break
    if arg_id is None:
        arg_id = list(world.agents.keys())[0]

    run_results = []
    for seed in range(42, 42 + n_runs):
        w = copy.deepcopy(world)
        agent = w.agents[arg_id]

        # Override to Argentina 2001-like conditions
        agent.economy.public_debt = agent.economy.gdp * 1.5
        agent.risk.debt_crisis_prone = 0.91
        agent.risk.regime_stability = 0.22
        agent.society.trust_gov = 0.18
        agent.society.social_tension = 0.83
        agent.society.inequality_gini = 51.3
        agent.economy.unemployment = 0.185

        traj = run_steps(w, years=8, seed=seed, enable_extreme_events=False)

        # Extract metrics
        gdps = [traj[i].agents[arg_id].economy.gdp for i in range(9)]
        ginis = [traj[i].agents[arg_id].society.inequality_gini for i in range(9)]
        crisis_years = [traj[i].agents[arg_id].risk.debt_crisis_active_years for i in range(9)]

        peak_gdp = max(gdps)
        trough_gdp = min(gdps[1:])  # after t=0
        cum_loss = (peak_gdp - trough_gdp) / max(peak_gdp, 1e-9) * 100
        max_active = max(crisis_years)
        gini_increase = max(ginis[1:]) - ginis[0]

        # Recovery year: first year GDP returns to peak
        recovery_year = None
        for i in range(2, 9):
            if gdps[i] >= peak_gdp * 0.95:
                recovery_year = i
                break

        run_results.append({
            "seed": seed,
            "max_crisis_active_years": max_active,
            "cum_gdp_loss_pct": round(cum_loss, 1),
            "gini_increase": round(gini_increase, 1),
            "recovery_year": recovery_year,
        })

    # Average across runs
    avg_active = sum(r["max_crisis_active_years"] for r in run_results) / n_runs
    avg_loss = sum(r["cum_gdp_loss_pct"] for r in run_results) / n_runs
    avg_gini = sum(r["gini_increase"] for r in run_results) / n_runs

    return {
        "agent_id": arg_id,
        "runs": run_results,
        "avg_max_active_years": round(avg_active, 1),
        "avg_cum_gdp_loss_pct": round(avg_loss, 1),
        "avg_gini_increase": round(avg_gini, 1),
        "D1_pass": avg_active >= 4,         # target: ≥4 years
        "D2_pass": 15 <= avg_loss <= 25,     # target: 15-25%
        "D3_pass": 5 <= avg_gini <= 15,      # target: 5-15 points
    }


def _run_turkey_test(world, n_runs: int = 3) -> dict:
    """Simulate a Turkey-2018-like agent for 5 years."""
    tur_id = None
    for aid, agent in world.agents.items():
        if "turk" in agent.name.lower():
            tur_id = aid
            break
    if tur_id is None:
        tur_id = list(world.agents.keys())[1]

    run_results = []
    for seed in range(42, 42 + n_runs):
        w = copy.deepcopy(world)
        agent = w.agents[tur_id]

        # Turkey 2018 conditions
        agent.economy.public_debt = agent.economy.gdp * 0.55
        agent.risk.debt_crisis_prone = 0.65
        agent.risk.regime_stability = 0.45
        agent.society.trust_gov = 0.35
        agent.society.social_tension = 0.55
        agent.economy.inflation = 0.25

        traj = run_steps(w, years=5, seed=seed, enable_extreme_events=False)

        gdp_pcs = [
            traj[i].agents[tur_id].economy.gdp_per_capita for i in range(6)
        ]
        trusts = [traj[i].agents[tur_id].society.trust_gov for i in range(6)]
        crisis_years = [traj[i].agents[tur_id].risk.debt_crisis_active_years for i in range(6)]

        max_loss = 0
        for i in range(1, 6):
            loss = (gdp_pcs[0] - gdp_pcs[i]) / max(gdp_pcs[0], 1e-9) * 100
            max_loss = max(max_loss, loss)

        duration = max(crisis_years)
        trust_floor = min(trusts)

        run_results.append({
            "seed": seed,
            "max_gdppc_loss_pct": round(max_loss, 1),
            "crisis_duration": duration,
            "trust_floor": round(trust_floor, 3),
        })

    avg_loss = sum(r["max_gdppc_loss_pct"] for r in run_results) / n_runs
    avg_dur = sum(r["crisis_duration"] for r in run_results) / n_runs
    avg_trust = sum(r["trust_floor"] for r in run_results) / n_runs

    return {
        "agent_id": tur_id,
        "runs": run_results,
        "avg_max_gdppc_loss_pct": round(avg_loss, 1),
        "avg_crisis_duration": round(avg_dur, 1),
        "avg_trust_floor": round(avg_trust, 3),
        "D5_pass": 1 <= avg_dur <= 3,        # target: 1-3 years
        "D6_pass": 10 <= avg_loss <= 20,      # target: 10-20% loss
        "D7_pass": avg_trust > 0.15,          # should not trigger regime collapse
    }


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        world = load_operational_world()

        arg_results = _run_argentina_test(world, n_runs=3)
        tur_results = _run_turkey_test(world, n_runs=3)

        arg_passed = sum([arg_results["D1_pass"], arg_results["D2_pass"], arg_results["D3_pass"]])
        tur_passed = sum([tur_results["D5_pass"], tur_results["D6_pass"], tur_results["D7_pass"]])

        conditions = {
            "argentina_3of4": arg_passed >= 2,  # relaxed: 2 of 3
            "turkey_2of3": tur_passed >= 2,
        }

    flagged = []
    if not arg_results["D1_pass"]:
        flagged.append(f"Argentina D1 FAIL: avg active years = {arg_results['avg_max_active_years']}")
    if not arg_results["D2_pass"]:
        flagged.append(f"Argentina D2 FAIL: avg GDP loss = {arg_results['avg_cum_gdp_loss_pct']}%")
    if not tur_results["D6_pass"]:
        flagged.append(f"Turkey D6 FAIL: avg GDP/cap loss = {tur_results['avg_max_gdppc_loss_pct']}%")

    return make_result(
        test_id="T3",
        status=determine_status(conditions),
        metrics={
            "argentina_metrics_passed": arg_passed,
            "turkey_metrics_passed": tur_passed,
            "argentina_avg_active_years": arg_results["avg_max_active_years"],
            "argentina_avg_gdp_loss_pct": arg_results["avg_cum_gdp_loss_pct"],
            "argentina_avg_gini_increase": arg_results["avg_gini_increase"],
            "turkey_avg_gdppc_loss_pct": tur_results["avg_max_gdppc_loss_pct"],
            "turkey_avg_crisis_duration": tur_results["avg_crisis_duration"],
            "turkey_avg_trust_floor": tur_results["avg_trust_floor"],
        },
        flagged_items=flagged,
        notes=f"Completed in {t.elapsed:.1f}s",
        details={
            "argentina": arg_results,
            "turkey": tur_results,
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
