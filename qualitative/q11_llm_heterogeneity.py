"""
Q11 — LLM Behavioral Heterogeneity Under Full Simulation Trajectory

Run a 3-year LLM-driven simulation on 10 agents, partition agents into
archetypes, and measure whether structurally different countries pursue
distinguishable policy trajectories.
"""
from __future__ import annotations
import statistics
from ..helpers import (
    gim, load_compact_world, run_steps_llm,
    make_result, determine_status, Timer,
)


CLIMATE_ORDER = {"none": 0, "weak": 1, "moderate": 2, "strong": 3}


def _classify_archetype(agent) -> str:
    """Classify agent into archetype: high_income_demo, middle_hybrid, low_autocratic."""
    gdp_pc = agent.economy.gdp_per_capita
    regime = agent.culture.regime_type.lower()
    if gdp_pc > 30000 and "demo" in regime:
        return "high_income_demo"
    elif gdp_pc < 8000 or "auto" in regime:
        return "low_autocratic"
    else:
        return "middle_hybrid"


def _action_profile(action_records: list[dict], agent_id: str) -> dict:
    """Build an aggregate action profile from the action log."""
    agent_rows = [r for r in action_records if r.get("agent_id") == agent_id]
    if not agent_rows:
        return {"climate_dist": {}, "avg_mil": 0, "avg_social": 0,
                "sanctions_count": 0, "trade_deals_count": 0}

    climate_counts = {"none": 0, "weak": 0, "moderate": 0, "strong": 0}
    mil_changes = []
    social_changes = []
    sanctions = 0
    trade_deals = 0
    for r in agent_rows:
        cp = r.get("dom_climate_policy", "none")
        climate_counts[cp] = climate_counts.get(cp, 0) + 1
        mil_changes.append(float(r.get("dom_military_spending_change", 0)))
        social_changes.append(float(r.get("dom_social_spending_change", 0)))
        # Count sanctions (from JSON string)
        import json
        try:
            s = json.loads(r.get("sanctions_intent", "[]"))
            sanctions += len([x for x in s if x.get("type", "none") != "none"])
        except Exception:
            pass
        try:
            td = json.loads(r.get("trade_deals", "[]"))
            trade_deals += len(td)
        except Exception:
            pass

    n = len(agent_rows)
    return {
        "climate_dist": {k: v / n for k, v in climate_counts.items()},
        "avg_mil": statistics.mean(mil_changes),
        "avg_social": statistics.mean(social_changes),
        "sanctions_per_year": sanctions / max(n, 1),
        "trade_deals_per_year": trade_deals / max(n, 1),
    }


def _profile_to_vector(profile: dict) -> list[float]:
    """Flatten profile to a numeric vector for distance computation."""
    cd = profile.get("climate_dist", {})
    return [
        cd.get("none", 0), cd.get("weak", 0), cd.get("moderate", 0), cd.get("strong", 0),
        profile.get("avg_mil", 0) * 100,     # scale to comparable range
        profile.get("avg_social", 0) * 100,
        profile.get("sanctions_per_year", 0),
        profile.get("trade_deals_per_year", 0) / 4,  # normalize
    ]


def _l1(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        world = load_compact_world()
        action_log: list[dict] = []

        traj = run_steps_llm(world, years=3, seed=42, action_log=action_log)

        # Classify agents
        archetypes: dict[str, list[str]] = {}
        for aid, agent in world.agents.items():
            arch = _classify_archetype(agent)
            archetypes.setdefault(arch, []).append(aid)

        # Build per-archetype centroid profiles
        arch_profiles: dict[str, list[float]] = {}
        agent_profiles: dict[str, list[float]] = {}
        for arch, aids in archetypes.items():
            vectors = []
            for aid in aids:
                profile = _action_profile(action_log, aid)
                vec = _profile_to_vector(profile)
                agent_profiles[aid] = vec
                vectors.append(vec)
            if vectors:
                centroid = [statistics.mean(col) for col in zip(*vectors)]
                arch_profiles[arch] = centroid

        # Inter-archetype distance (mean pairwise L1 between centroids)
        arch_names = list(arch_profiles.keys())
        inter_distances = []
        for i in range(len(arch_names)):
            for j in range(i + 1, len(arch_names)):
                inter_distances.append(
                    _l1(arch_profiles[arch_names[i]], arch_profiles[arch_names[j]])
                )
        mean_inter = statistics.mean(inter_distances) if inter_distances else 0

        # Intra-archetype distance (mean L1 from each agent to its centroid)
        intra_distances = []
        for arch, aids in archetypes.items():
            centroid = arch_profiles.get(arch)
            if centroid is None:
                continue
            for aid in aids:
                intra_distances.append(_l1(agent_profiles.get(aid, centroid), centroid))
        mean_intra = statistics.mean(intra_distances) if intra_distances else 1e-6

        discrimination_ratio = mean_inter / max(mean_intra, 1e-6)

        # Check modal climate policy per archetype
        arch_modal_climate = {}
        for arch, aids in archetypes.items():
            climate_totals = {"none": 0, "weak": 0, "moderate": 0, "strong": 0}
            for aid in aids:
                profile = _action_profile(action_log, aid)
                for cp, frac in profile.get("climate_dist", {}).items():
                    climate_totals[cp] = climate_totals.get(cp, 0) + frac
            modal = max(climate_totals, key=climate_totals.get) if climate_totals else "none"
            arch_modal_climate[arch] = modal

        distinct_climate = len(set(arch_modal_climate.values()))

        conditions = {
            "discrimination_ratio_gt_1.5": discrimination_ratio > 1.5,
            "distinct_climate_ge_2": distinct_climate >= 2,
        }

    return make_result(
        test_id="Q11",
        status=determine_status(conditions),
        metrics={
            "mean_inter_archetype_distance": round(mean_inter, 4),
            "mean_intra_archetype_distance": round(mean_intra, 4),
            "discrimination_ratio": round(discrimination_ratio, 3),
            "distinct_modal_climate_policies": distinct_climate,
            "archetype_sizes": {k: len(v) for k, v in archetypes.items()},
            "archetype_modal_climate": arch_modal_climate,
            "total_action_log_rows": len(action_log),
        },
        flagged_items=[
            f"Discrimination ratio = {discrimination_ratio:.2f} (threshold: >1.5)"
            if discrimination_ratio <= 1.5 else "",
        ],
        notes=f"Completed in {t.elapsed:.1f}s. {len(action_log)} action-log rows.",
        details={
            "archetype_centroids": {k: [round(x, 4) for x in v] for k, v in arch_profiles.items()},
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
