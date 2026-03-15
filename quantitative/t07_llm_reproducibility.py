"""
T7 — LLM Reproducibility: Trajectory Stability Across Identical Runs

Run the same 3-year LLM simulation twice with identical state and seed,
measure macro-trajectory divergence caused by LLM output stochasticity.
"""
from __future__ import annotations
import json
import statistics
from ..helpers import (
    gim, load_compact_world, run_steps_llm,
    make_result, determine_status, Timer,
)


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        world = load_compact_world()

        # ------------------------------------------------------------------
        # Run A
        # ------------------------------------------------------------------
        action_log_a: list[dict] = []
        traj_a = run_steps_llm(world, years=3, seed=42, action_log=action_log_a)

        # ------------------------------------------------------------------
        # Run B (identical inputs)
        # ------------------------------------------------------------------
        action_log_b: list[dict] = []
        traj_b = run_steps_llm(world, years=3, seed=42, action_log=action_log_b)

        # ------------------------------------------------------------------
        # Compute per-agent divergence at t=3
        # ------------------------------------------------------------------
        gdp_divs = []
        trust_divs = []
        tension_divs = []

        agent_details = []
        for aid in world.agents:
            ga = traj_a[3].agents[aid].economy.gdp
            gb = traj_b[3].agents[aid].economy.gdp
            mean_g = (ga + gb) / 2
            gdp_div = abs(ga - gb) / max(mean_g, 1e-9)
            gdp_divs.append(gdp_div)

            ta = traj_a[3].agents[aid].society.trust_gov
            tb = traj_b[3].agents[aid].society.trust_gov
            trust_div = abs(ta - tb)
            trust_divs.append(trust_div)

            tna = traj_a[3].agents[aid].society.social_tension
            tnb = traj_b[3].agents[aid].society.social_tension
            tension_div = abs(tna - tnb)
            tension_divs.append(tension_div)

            agent_details.append({
                "agent_id": aid,
                "name": world.agents[aid].name,
                "gdp_divergence": round(gdp_div, 5),
                "trust_divergence": round(trust_div, 5),
                "tension_divergence": round(tension_div, 5),
            })

        mean_gdp_div = statistics.mean(gdp_divs) if gdp_divs else 0
        mean_trust_div = statistics.mean(trust_divs) if trust_divs else 0
        mean_tension_div = statistics.mean(tension_divs) if tension_divs else 0

        # ------------------------------------------------------------------
        # Compare action labels
        # ------------------------------------------------------------------
        def _extract_action_key(log_row: dict) -> str:
            """Build a rough action fingerprint from the action log row."""
            parts = []
            cp = log_row.get("dom_climate_policy", "none")
            if cp != "none":
                parts.append(f"climate_{cp}")
            sec = log_row.get("security_applied_type", "none")
            if sec != "none":
                parts.append(sec)
            try:
                sanctions = json.loads(log_row.get("sanctions_intent", "[]"))
                if any(s.get("type", "none") != "none" for s in sanctions):
                    parts.append("sanctions")
            except Exception:
                pass
            mil = float(log_row.get("dom_military_spending_change", 0))
            if mil > 0.003:
                parts.append("mil_up")
            elif mil < -0.003:
                parts.append("mil_down")
            return "+".join(parts) if parts else "status_quo"

        # Build (agent, step) -> action_key maps
        actions_a: dict[tuple[str, int], str] = {}
        actions_b: dict[tuple[str, int], str] = {}
        for row in action_log_a:
            key = (row["agent_id"], row["time"])
            actions_a[key] = _extract_action_key(row)
        for row in action_log_b:
            key = (row["agent_id"], row["time"])
            actions_b[key] = _extract_action_key(row)

        all_keys = set(actions_a.keys()) | set(actions_b.keys())
        matched = sum(1 for k in all_keys if actions_a.get(k) == actions_b.get(k))
        action_match_rate = matched / max(len(all_keys), 1)

        # ------------------------------------------------------------------
        # Assess
        # ------------------------------------------------------------------
        conditions = {
            "mean_gdp_div_lt_008": mean_gdp_div < 0.08,
            "mean_trust_div_lt_010": mean_trust_div < 0.10,
            "action_match_rate_gt_060": action_match_rate > 0.60,
        }

    return make_result(
        test_id="T7",
        status=determine_status(conditions),
        metrics={
            "mean_gdp_divergence": round(mean_gdp_div, 5),
            "max_gdp_divergence": round(max(gdp_divs), 5) if gdp_divs else 0,
            "mean_trust_divergence": round(mean_trust_div, 5),
            "mean_tension_divergence": round(mean_tension_div, 5),
            "action_match_rate": round(action_match_rate, 3),
            "matched_action_steps": matched,
            "total_action_steps": len(all_keys),
        },
        flagged_items=[
            f"Mean GDP divergence = {mean_gdp_div:.3%} (threshold: <8%)"
            if mean_gdp_div >= 0.08 else "",
            f"Action match rate = {action_match_rate:.0%} (threshold: >60%)"
            if action_match_rate <= 0.60 else "",
        ],
        notes=f"Completed in {t.elapsed:.1f}s. 2 runs × 30 LLM calls = ~60 calls.",
        details={
            "agent_details": agent_details,
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
