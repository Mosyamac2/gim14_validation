"""
Q3 — LLM Policy Prompt Structural Homogeneity and Behavioral Convergence Risk

Construct synthetic archetype observations and check whether the fixed
prompt template can elicit distinguishable policy vectors.  Optionally
calls the LLM API if DEEPSEEK_API_KEY is set.
"""
from __future__ import annotations
import json
import os
from ..helpers import gim, load_operational_world, make_result, determine_status, Timer


ARCHETYPES = {
    "wealthy_democracy": {
        "trust_gov": 0.75, "social_tension": 0.15, "inequality_gini": 30,
        "gdp": 20.0, "gdp_per_capita": 65000, "climate_risk": 0.25,
        "regime_type": "Democracy", "pdi": 35, "idv": 85, "mas": 60,
        "conflict_proneness": 0.15, "regime_stability": 0.9,
        "alliance_block": "Western", "co2_annual_emissions": 4.5,
        "security_margin": 1.5,
    },
    "wealthy_autocracy": {
        "trust_gov": 0.40, "social_tension": 0.50, "inequality_gini": 45,
        "gdp": 3.5, "gdp_per_capita": 25000, "climate_risk": 0.45,
        "regime_type": "Autocracy", "pdi": 80, "idv": 25, "mas": 55,
        "conflict_proneness": 0.55, "regime_stability": 0.6,
        "alliance_block": "Eurasian", "co2_annual_emissions": 1.8,
        "security_margin": 2.0,
    },
    "poor_fragile": {
        "trust_gov": 0.20, "social_tension": 0.75, "inequality_gini": 55,
        "gdp": 0.05, "gdp_per_capita": 800, "climate_risk": 0.70,
        "regime_type": "HybridRegime", "pdi": 70, "idv": 20, "mas": 40,
        "conflict_proneness": 0.70, "regime_stability": 0.25,
        "alliance_block": "NonAligned", "co2_annual_emissions": 0.02,
        "security_margin": 0.6,
    },
    "resource_rich_mena": {
        "trust_gov": 0.55, "social_tension": 0.35, "inequality_gini": 40,
        "gdp": 1.0, "gdp_per_capita": 22000, "climate_risk": 0.55,
        "regime_type": "Autocracy", "pdi": 85, "idv": 30, "mas": 50,
        "conflict_proneness": 0.45, "regime_stability": 0.55,
        "alliance_block": "MENA", "co2_annual_emissions": 0.6,
        "security_margin": 1.1,
    },
    "rising_power": {
        "trust_gov": 0.60, "social_tension": 0.30, "inequality_gini": 38,
        "gdp": 3.0, "gdp_per_capita": 12000, "climate_risk": 0.40,
        "regime_type": "HybridRegime", "pdi": 65, "idv": 40, "mas": 55,
        "conflict_proneness": 0.35, "regime_stability": 0.65,
        "alliance_block": "NonAligned", "co2_annual_emissions": 2.5,
        "security_margin": 1.3,
    },
    "small_open_economy": {
        "trust_gov": 0.80, "social_tension": 0.10, "inequality_gini": 28,
        "gdp": 0.4, "gdp_per_capita": 55000, "climate_risk": 0.15,
        "regime_type": "Democracy", "pdi": 30, "idv": 75, "mas": 15,
        "conflict_proneness": 0.10, "regime_stability": 0.92,
        "alliance_block": "Western", "co2_annual_emissions": 0.04,
        "security_margin": 0.7,
    },
}


def _predict_heuristic_actions(archetype: dict) -> dict:
    """Predict action from the explicit prompt heuristics."""
    # Climate policy heuristic from prompt
    if archetype["gdp_per_capita"] < 5000 or archetype["social_tension"] > 0.6:
        climate = "none"
    elif archetype["climate_risk"] > 0.4 and archetype["gdp_per_capita"] > 20000:
        climate = "moderate"
    else:
        climate = "weak"

    # Security action heuristic from prompt thresholds
    if archetype["conflict_proneness"] > 0.80 and archetype["trust_gov"] < 0.15:
        security = "conflict"
    elif archetype["conflict_proneness"] > 0.60 and archetype["trust_gov"] < 0.30:
        security = "border_incident"
    elif archetype["conflict_proneness"] > 0.35:
        security = "military_exercise"
    else:
        security = "none"

    # Military spending direction
    if archetype["security_margin"] < 1.0:
        mil_direction = "increase"
    else:
        mil_direction = "flat_or_decrease"

    return {
        "climate_policy": climate,
        "security_action": security,
        "mil_spending_direction": mil_direction,
    }


def _actions_differ(a1: dict, a2: dict) -> bool:
    """Check if two predicted action profiles are qualitatively distinct."""
    return (
        a1["climate_policy"] != a2["climate_policy"]
        or a1["security_action"] != a2["security_action"]
        or a1["mil_spending_direction"] != a2["mil_spending_direction"]
    )


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        # ------------------------------------------------------------------
        # Heuristic prediction: trace prompt rules on each archetype
        # ------------------------------------------------------------------
        predicted_actions = {}
        for name, arch in ARCHETYPES.items():
            predicted_actions[name] = _predict_heuristic_actions(arch)

        # Count distinct profiles
        profiles = list(predicted_actions.values())
        distinct_count = 0
        for i, p1 in enumerate(profiles):
            is_unique = True
            for j in range(i):
                if not _actions_differ(p1, profiles[j]):
                    is_unique = False
                    break
            if is_unique:
                distinct_count += 1

        heuristic_diverse = distinct_count >= 4

        # ------------------------------------------------------------------
        # Optional LLM check (only if DEEPSEEK_API_KEY is available)
        # ------------------------------------------------------------------
        llm_results = None
        llm_diverse = None
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if api_key and verbose:
            try:
                from gim.core.policy import call_llm, LLM_POLICY_PROMPT_TEMPLATE, LLM_SCHEMA_HINT
                llm_results = {}
                for name, arch in ARCHETYPES.items():
                    obs_json = json.dumps({"self_state": arch, "agent_id": name, "time": 0,
                                           "resource_balance": {}, "external_actors": {}})
                    prompt = LLM_POLICY_PROMPT_TEMPLATE.format(
                        obs_json=obs_json, schema_hint=LLM_SCHEMA_HINT)
                    raw = call_llm(prompt)
                    start = raw.find("{")
                    end = raw.rfind("}")
                    if start >= 0 and end > start:
                        data = json.loads(raw[start:end+1])
                        llm_results[name] = {
                            "climate_policy": data.get("domestic_policy", {}).get("climate_policy"),
                            "security_type": data.get("foreign_policy", {}).get("security_actions", {}).get("type"),
                        }
            except Exception as exc:
                llm_results = {"error": str(exc)}

        conditions = {
            "heuristic_distinct_ge_4": heuristic_diverse,
        }

    return make_result(
        test_id="Q3",
        status=determine_status(conditions),
        metrics={
            "distinct_heuristic_profiles": distinct_count,
            "total_archetypes": len(ARCHETYPES),
        },
        flagged_items=[
            f"Only {distinct_count} distinct profiles from {len(ARCHETYPES)} archetypes"
            if not heuristic_diverse else "",
        ],
        notes=f"Completed in {t.elapsed:.1f}s. LLM {'called' if llm_results else 'skipped'}.",
        details={
            "predicted_actions": predicted_actions,
            "llm_results": llm_results,
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
