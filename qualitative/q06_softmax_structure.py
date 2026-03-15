"""
Q6 — Scenario Outcome Model: Softmax Over Linear Scores with Expert Priors

Analyze the softmax-over-linear-scores architecture for structural bias
toward status_quo and lack of interaction terms.
"""
from __future__ import annotations
import math
from ..helpers import gim, load_operational_world, make_result, determine_status, Timer


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    max_s = max(scores.values())
    exps = {k: math.exp(v - max_s) for k, v in scores.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        from gim.geo_calibration import (
            OUTCOME_INTERCEPTS, OUTCOME_DRIVERS, lookup_value,
        )

        # ------------------------------------------------------------------
        # Step 1: Baseline softmax with only intercepts (all drivers = 0)
        # ------------------------------------------------------------------
        intercept_scores = {k: w.value for k, w in OUTCOME_INTERCEPTS.items()}
        baseline_probs = _softmax(intercept_scores)
        sq_baseline = baseline_probs.get("status_quo", 0)
        crisis_outcomes = [
            "controlled_suppression", "internal_destabilization",
            "limited_proxy_escalation", "maritime_chokepoint_crisis",
            "direct_strike_exchange", "broad_regional_escalation",
        ]
        crisis_baseline = sum(baseline_probs.get(k, 0) for k in crisis_outcomes)

        # ------------------------------------------------------------------
        # Step 2: All drivers at 75th percentile stress
        # ------------------------------------------------------------------
        # Simulate high-stress scenario: all positive-risk drivers at 0.75
        compound_scores = dict(intercept_scores)
        for outcome, drivers in OUTCOME_DRIVERS.items():
            for driver_name, weight in drivers.items():
                # Positive weights = stress pushers; use 0.75
                # Negative weights = stability factors; use 0.25
                driver_val = 0.75 if weight.value > 0 else 0.25
                compound_scores[outcome] = compound_scores.get(outcome, 0) + weight.value * driver_val

        compound_probs = _softmax(compound_scores)
        crisis_compound = sum(compound_probs.get(k, 0) for k in crisis_outcomes)

        # ------------------------------------------------------------------
        # Step 3: Single extreme driver (conflict_stress at 99th pctile = 0.99)
        # ------------------------------------------------------------------
        single_scores = dict(intercept_scores)
        for outcome, drivers in OUTCOME_DRIVERS.items():
            if "conflict_stress" in drivers:
                single_scores[outcome] += drivers["conflict_stress"].value * 0.99

        single_probs = _softmax(single_scores)
        crisis_single = sum(single_probs.get(k, 0) for k in crisis_outcomes)

        # ------------------------------------------------------------------
        # Assess
        # ------------------------------------------------------------------
        # Compound should produce at least as much crisis mass as single extreme
        compound_ge_single = crisis_compound >= crisis_single
        # Baseline status_quo should be < 60%
        sq_not_dominant = sq_baseline < 0.60

        conditions = {
            "compound_ge_single_crisis_mass": compound_ge_single,
            "baseline_sq_lt_60pct": sq_not_dominant,
        }

    return make_result(
        test_id="Q6",
        status=determine_status(conditions),
        metrics={
            "baseline_status_quo_pct": round(sq_baseline * 100, 2),
            "baseline_crisis_mass_pct": round(crisis_baseline * 100, 2),
            "compound_crisis_mass_pct": round(crisis_compound * 100, 2),
            "single_extreme_crisis_mass_pct": round(crisis_single * 100, 2),
            "intercept_gap_sq_vs_broad": round(
                intercept_scores["status_quo"] - intercept_scores["broad_regional_escalation"], 3
            ),
        },
        flagged_items=[
            f"Baseline status_quo = {sq_baseline*100:.1f}% (threshold: <60%)"
            if not sq_not_dominant else "",
            f"Compound crisis mass {crisis_compound*100:.1f}% < single-driver {crisis_single*100:.1f}%"
            if not compound_ge_single else "",
        ],
        notes=f"Completed in {t.elapsed:.1f}s",
        details={
            "baseline_probs": {k: round(v, 4) for k, v in baseline_probs.items()},
            "compound_probs": {k: round(v, 4) for k, v in compound_probs.items()},
            "single_extreme_probs": {k: round(v, 4) for k, v in single_probs.items()},
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
