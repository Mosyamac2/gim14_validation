"""
T8 — LLM Policy Coherence: Context-Appropriate Actions

Verify that LLM-generated policies are contextually coherent:
stressed agents adopt crisis-appropriate actions, stable agents
don't randomly escalate.
"""
from __future__ import annotations
import json
from ..helpers import (
    gim, load_compact_world, run_steps_llm,
    make_result, determine_status, Timer,
)


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        world = load_compact_world()
        action_log: list[dict] = []

        traj = run_steps_llm(world, years=3, seed=42, action_log=action_log)

        violations: list[dict] = []
        applicable: dict[str, int] = {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0}
        violated: dict[str, int] = {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0}

        # Build helper to look up agent state at a given step
        def _agent_at(aid: str, step: int):
            return traj[step].agents[aid]

        for row in action_log:
            aid = row["agent_id"]
            step = row["time"]
            agent = _agent_at(aid, min(step + 1, len(traj) - 1))
            agent_prev = _agent_at(aid, step)
            name = row.get("agent_name", aid)

            tension = agent_prev.society.social_tension
            trust = agent_prev.society.trust_gov
            debt_gdp = agent_prev.economy.public_debt / max(agent_prev.economy.gdp, 1e-9)
            gdp_pc = agent_prev.economy.gdp_per_capita
            climate_risk = agent_prev.climate.climate_risk

            mil_change = float(row.get("dom_military_spending_change", 0))
            social_change = float(row.get("dom_social_spending_change", 0))
            climate_policy = row.get("dom_climate_policy", "none")
            explanation = row.get("explanation", "")

            # Security margin
            from gim.core.metrics import compute_relative_metrics
            security_margin = getattr(agent_prev, "_security_margin", None)
            # Approximate: use military_power vs neighbors
            own_mil = agent_prev.technology.military_power
            neighbor_mils = [
                traj[step].agents[nid].technology.military_power
                for nid in traj[step].relations.get(aid, {})
                if nid in traj[step].agents
            ]
            avg_neighbor_mil = (sum(neighbor_mils) / len(neighbor_mils)) if neighbor_mils else 1.0
            sec_margin = own_mil / max(avg_neighbor_mil, 1e-6)

            max_neighbor_conflict = max(
                (traj[step].relations.get(aid, {}).get(nid).conflict_level
                 for nid in traj[step].relations.get(aid, {})
                 if nid in traj[step].agents),
                default=0,
            )

            # ---------------------------------------------------------------
            # C1: Crisis response — stressed agents shouldn't boost military
            # ---------------------------------------------------------------
            debt_stress = max(0, debt_gdp - 1.0) * agent_prev.risk.debt_crisis_prone
            if debt_stress > 0.5 or tension > 0.6:
                applicable["C1"] += 1
                if mil_change > 0.005:
                    violated["C1"] += 1
                    violations.append({
                        "rule": "C1", "step": step, "agent": name,
                        "detail": f"mil_change={mil_change:.4f} despite debt_stress={debt_stress:.2f}/tension={tension:.2f}",
                    })

            # ---------------------------------------------------------------
            # C2: Security awareness — threatened agents shouldn't cut military
            # ---------------------------------------------------------------
            if sec_margin < 1.0 and max_neighbor_conflict > 0.5:
                applicable["C2"] += 1
                sec_type = row.get("security_applied_type", "none")
                if mil_change < -0.001 and sec_type == "none":
                    violated["C2"] += 1
                    violations.append({
                        "rule": "C2", "step": step, "agent": name,
                        "detail": f"mil cut with sec_margin={sec_margin:.2f}, neighbor conflict={max_neighbor_conflict:.2f}",
                    })

            # ---------------------------------------------------------------
            # C3: Climate-income alignment
            # ---------------------------------------------------------------
            if gdp_pc < 5000:
                applicable["C3"] += 1
                if climate_policy == "strong":
                    violated["C3"] += 1
                    violations.append({
                        "rule": "C3", "step": step, "agent": name,
                        "detail": f"Strong climate policy with gdp_pc={gdp_pc:.0f}",
                    })

            if gdp_pc > 40000 and climate_risk > 0.3:
                applicable["C3"] += 1
                if climate_policy == "none":
                    violated["C3"] += 1
                    violations.append({
                        "rule": "C3", "step": step, "agent": name,
                        "detail": f"No climate policy with gdp_pc={gdp_pc:.0f}, risk={climate_risk:.2f}",
                    })

            # ---------------------------------------------------------------
            # C4: No self-sanction or ally-sanction without cause
            # ---------------------------------------------------------------
            try:
                sanctions = json.loads(row.get("sanctions_intent", "[]"))
                for s in sanctions:
                    target = s.get("target", "")
                    if s.get("type", "none") == "none":
                        continue
                    applicable["C4"] += 1
                    if target == aid:
                        violated["C4"] += 1
                        violations.append({
                            "rule": "C4", "step": step, "agent": name,
                            "detail": "Self-sanction",
                        })
                    # Check ally sanction
                    target_agent = traj[step].agents.get(target)
                    if (target_agent
                            and target_agent.alliance_block == agent_prev.alliance_block
                            and agent_prev.alliance_block != "NonAligned"):
                        rel = traj[step].relations.get(aid, {}).get(target)
                        if rel and rel.conflict_level < 0.5:
                            violated["C4"] += 1
                            violations.append({
                                "rule": "C4", "step": step, "agent": name,
                                "detail": f"Sanctioned ally {target_agent.name} (conflict={rel.conflict_level:.2f})",
                            })
            except Exception:
                pass

            # ---------------------------------------------------------------
            # C5: Explanation non-empty
            # ---------------------------------------------------------------
            applicable["C5"] += 1
            if not explanation or explanation.strip() == "" or explanation == "baseline do-nothing policy":
                violated["C5"] += 1
                violations.append({
                    "rule": "C5", "step": step, "agent": name,
                    "detail": "Empty or fallback explanation",
                })

        # Compute coherence score
        total_applicable = sum(applicable.values())
        total_violated = sum(violated.values())
        coherence_score = 1.0 - (total_violated / max(total_applicable, 1))

        # Per-rule violation rates
        rule_rates = {}
        for rule in applicable:
            if applicable[rule] > 0:
                rule_rates[rule] = round(violated[rule] / applicable[rule], 3)
            else:
                rule_rates[rule] = 0.0

        any_rule_gt_030 = any(r > 0.30 for r in rule_rates.values())

        conditions = {
            "coherence_score_gt_085": coherence_score > 0.85,
            "no_rule_gt_30pct": not any_rule_gt_030,
        }

    return make_result(
        test_id="T8",
        status=determine_status(conditions),
        metrics={
            "coherence_score": round(coherence_score, 3),
            "total_applicable": total_applicable,
            "total_violated": total_violated,
            "applicable_by_rule": applicable,
            "violated_by_rule": violated,
            "violation_rate_by_rule": rule_rates,
        },
        flagged_items=[
            f"Rule {rule}: {violated[rule]}/{applicable[rule]} violations ({rule_rates[rule]:.0%})"
            for rule in sorted(applicable)
            if rule_rates.get(rule, 0) > 0.15
        ],
        notes=f"Completed in {t.elapsed:.1f}s. {len(violations)} total violations.",
        details={
            "violations": violations[:30],
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
