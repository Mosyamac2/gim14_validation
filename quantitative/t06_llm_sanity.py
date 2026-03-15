"""
T6 — LLM Sanity Bounds: Plausible Trajectories Under LLM Policies

Run a 3-year LLM simulation on 10 agents and check that GDP, trust,
tension, debt, temperature, and population stay within plausible bounds.
"""
from __future__ import annotations
from ..helpers import (
    gim, load_compact_world, run_steps_llm,
    make_result, determine_status, Timer,
)


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        world = load_compact_world()
        action_log: list[dict] = []
        traj = run_steps_llm(world, years=3, seed=42, action_log=action_log)

        violations = []
        n_agents = len(world.agents)

        for step in range(1, 4):
            w_prev = traj[step - 1]
            w_curr = traj[step]
            for aid, agent in w_curr.agents.items():
                prev_agent = w_prev.agents[aid]
                name = agent.name

                # B1: GDP change > -20% per year (no single-year collapse without war)
                gdp_prev = max(prev_agent.economy.gdp, 1e-9)
                gdp_change = (agent.economy.gdp - gdp_prev) / gdp_prev
                at_war = any(
                    rel.at_war
                    for rel in w_curr.relations.get(aid, {}).values()
                )
                has_crisis = (
                    agent.risk.debt_crisis_active_years > 0
                    or agent.risk.regime_crisis_active_years > 0
                )
                if gdp_change < -0.20 and not at_war and not has_crisis:
                    violations.append({
                        "rule": "B1", "step": step, "agent": name,
                        "value": round(gdp_change, 4),
                        "detail": "GDP fell >20% without war/crisis",
                    })

                # B2: trust_gov in [0.05, 1.0]
                if agent.society.trust_gov < 0.05:
                    violations.append({
                        "rule": "B2", "step": step, "agent": name,
                        "value": round(agent.society.trust_gov, 4),
                        "detail": "Trust near machine-zero",
                    })

                # B3: social_tension in [0, 0.95] for initially stable agents
                was_stable = prev_agent.society.trust_gov > 0.5 and prev_agent.society.social_tension < 0.3
                if was_stable and agent.society.social_tension > 0.95:
                    violations.append({
                        "rule": "B3", "step": step, "agent": name,
                        "value": round(agent.society.social_tension, 4),
                        "detail": "Tension saturated for stable agent",
                    })

                # B4: debt/GDP < 3.0 unless debt crisis active
                debt_gdp = agent.economy.public_debt / max(agent.economy.gdp, 1e-9)
                if debt_gdp > 3.0 and agent.risk.debt_crisis_active_years == 0:
                    violations.append({
                        "rule": "B4", "step": step, "agent": name,
                        "value": round(debt_gdp, 4),
                        "detail": "Debt/GDP > 3.0 without active debt crisis",
                    })

                # B6: No population drop > 5% in a year (without war)
                pop_prev = max(prev_agent.economy.population, 1)
                pop_change = (agent.economy.population - pop_prev) / pop_prev
                if pop_change < -0.05 and not at_war:
                    violations.append({
                        "rule": "B6", "step": step, "agent": name,
                        "value": round(pop_change, 4),
                        "detail": "Population fell >5% without war",
                    })

            # B5: Global temperature change < 0.5°C over 3 years
            if step == 3:
                temp_change = abs(
                    w_curr.global_state.temperature_global
                    - traj[0].global_state.temperature_global
                )
                if temp_change > 0.5:
                    violations.append({
                        "rule": "B5", "step": step, "agent": "global",
                        "value": round(temp_change, 4),
                        "detail": "Temperature runaway from LLM policy",
                    })

        # Stable-agent B1 violations
        stable_b1 = [
            v for v in violations
            if v["rule"] == "B1"
            and world.agents.get(
                next((aid for aid, a in world.agents.items() if a.name == v["agent"]), ""), None
            ) is not None
            and world.agents[
                next((aid for aid, a in world.agents.items() if a.name == v["agent"]), "")
            ].society.trust_gov > 0.5
        ]

        total_violations = len(violations)
        conditions = {
            "total_violations_lt_3": total_violations < 3,
            "no_stable_b1": len(stable_b1) == 0,
        }

    return make_result(
        test_id="T6",
        status=determine_status(conditions),
        metrics={
            "total_violations": total_violations,
            "violations_by_rule": {
                rule: sum(1 for v in violations if v["rule"] == rule)
                for rule in ["B1", "B2", "B3", "B4", "B5", "B6"]
            },
            "n_agents": n_agents,
            "n_steps": 3,
            "total_agent_steps": n_agents * 3,
            "action_log_rows": len(action_log),
        },
        flagged_items=[
            f"{v['rule']} @ t={v['step']} {v['agent']}: {v['detail']} ({v['value']})"
            for v in violations[:10]
        ],
        notes=f"Completed in {t.elapsed:.1f}s.",
        details={
            "violations": violations,
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
