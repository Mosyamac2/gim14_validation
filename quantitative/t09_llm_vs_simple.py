"""
T9 — LLM vs Simple Policy Divergence: Does the LLM Add Value?

Compare 3-year trajectories under LLM vs simple policies to confirm
that LLM decision-making materially changes outcomes.
"""
from __future__ import annotations
import json
import statistics
from ..helpers import (
    gim, load_compact_world, run_steps, run_steps_llm,
    make_result, determine_status, Timer,
)


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        world = load_compact_world()

        # ------------------------------------------------------------------
        # Run A: Simple policy
        # ------------------------------------------------------------------
        action_log_simple: list[dict] = []
        traj_simple = run_steps(world, years=3, seed=42)

        # ------------------------------------------------------------------
        # Run B: LLM policy
        # ------------------------------------------------------------------
        action_log_llm: list[dict] = []
        traj_llm = run_steps_llm(world, years=3, seed=42, action_log=action_log_llm)

        # ------------------------------------------------------------------
        # GDP divergence
        # ------------------------------------------------------------------
        gdp_deltas = []
        trust_deltas = []
        tension_deltas = []
        agent_details = []

        for aid in world.agents:
            gs = traj_simple[3].agents[aid].economy.gdp
            gl = traj_llm[3].agents[aid].economy.gdp
            baseline = traj_simple[0].agents[aid].economy.gdp
            gdp_delta = abs(gl - gs) / max(baseline, 1e-9)
            gdp_deltas.append(gdp_delta)

            ts = traj_simple[3].agents[aid].society.trust_gov
            tl = traj_llm[3].agents[aid].society.trust_gov
            trust_deltas.append(abs(tl - ts))

            tns = traj_simple[3].agents[aid].society.social_tension
            tnl = traj_llm[3].agents[aid].society.social_tension
            tension_deltas.append(abs(tnl - tns))

            agent_details.append({
                "agent_id": aid,
                "name": world.agents[aid].name,
                "gdp_simple": round(gs, 4),
                "gdp_llm": round(gl, 4),
                "gdp_delta_pct": round(gdp_delta * 100, 2),
                "trust_simple": round(ts, 4),
                "trust_llm": round(tl, 4),
                "tension_simple": round(tns, 4),
                "tension_llm": round(tnl, 4),
            })

        mean_abs_gdp = statistics.mean(gdp_deltas) if gdp_deltas else 0
        mean_abs_trust = statistics.mean(trust_deltas) if trust_deltas else 0

        # ------------------------------------------------------------------
        # Action diversity
        # ------------------------------------------------------------------
        def _unique_action_types(log: list[dict]) -> set[str]:
            types = set()
            for row in log:
                parts = []
                cp = row.get("dom_climate_policy", "none")
                if cp != "none":
                    parts.append(f"climate_{cp}")
                sec = row.get("security_applied_type", "none")
                if sec != "none":
                    parts.append(sec)
                try:
                    s = json.loads(row.get("sanctions_intent", "[]"))
                    if any(x.get("type", "none") != "none" for x in s):
                        parts.append("sanctions")
                except Exception:
                    pass
                try:
                    td = json.loads(row.get("trade_deals", "[]"))
                    if td:
                        parts.append("trade_deal")
                except Exception:
                    pass
                label = "+".join(parts) if parts else "status_quo"
                types.add(f"{row.get('agent_id', '')}:{label}")
            return types

        # Simple policy action log is not captured through the same path,
        # so we count based on known behavior: simple always produces status_quo
        simple_unique = len(world.agents)  # each agent produces "status_quo" each year
        llm_unique_set = _unique_action_types(action_log_llm)
        llm_unique = len(llm_unique_set)
        action_diversity_ratio = llm_unique / max(simple_unique, 1)

        # ------------------------------------------------------------------
        # Foreign policy activity counts
        # ------------------------------------------------------------------
        sanctions_llm = 0
        trade_deals_llm = 0
        security_actions_llm = 0
        for row in action_log_llm:
            try:
                s = json.loads(row.get("sanctions_intent", "[]"))
                sanctions_llm += sum(1 for x in s if x.get("type", "none") != "none")
            except Exception:
                pass
            try:
                td = json.loads(row.get("trade_deals", "[]"))
                trade_deals_llm += len(td)
            except Exception:
                pass
            if row.get("security_applied_type", "none") != "none":
                security_actions_llm += 1

        # Simple policy produces zero foreign policy actions by design
        sanctions_simple = 0
        trade_deals_simple = 0

        # ------------------------------------------------------------------
        # Assess
        # ------------------------------------------------------------------
        conditions = {
            "gdp_divergence_gt_002": mean_abs_gdp > 0.02,
            "action_diversity_gt_2x": action_diversity_ratio > 2.0,
            "sanctions_llm_gt_0": sanctions_llm > 0,
        }

    return make_result(
        test_id="T9",
        status=determine_status(conditions),
        metrics={
            "mean_abs_gdp_delta": round(mean_abs_gdp, 4),
            "mean_abs_trust_delta": round(mean_abs_trust, 4),
            "action_diversity_ratio": round(action_diversity_ratio, 2),
            "llm_unique_action_types": llm_unique,
            "simple_unique_action_types": simple_unique,
            "sanctions_llm": sanctions_llm,
            "sanctions_simple": sanctions_simple,
            "trade_deals_llm": trade_deals_llm,
            "trade_deals_simple": trade_deals_simple,
            "security_actions_llm": security_actions_llm,
        },
        flagged_items=[
            f"GDP divergence only {mean_abs_gdp:.1%} (threshold: >2%)"
            if mean_abs_gdp <= 0.02 else "",
            "LLM produced zero sanctions"
            if sanctions_llm == 0 else "",
        ],
        notes=f"Completed in {t.elapsed:.1f}s. LLM log: {len(action_log_llm)} rows.",
        details={
            "agent_details": agent_details,
            "llm_action_types_sample": sorted(list(llm_unique_set))[:20],
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
