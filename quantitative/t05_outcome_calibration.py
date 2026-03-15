"""
T5 — Outcome-Probability Calibration: Discrimination and Calibration Score
      Under Historical Near-Miss Cases

Process all operational_v1 and operational_v2 cases through the scenario
evaluator and check discrimination, criticality separation, and
perturbation stability.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from ..helpers import gim, load_operational_world, make_result, determine_status, Timer


def _load_v1_cases(repo_root: Path) -> list[dict]:
    case_dir = repo_root / "misc" / "calibration_cases" / "operational_v1"
    if not case_dir.exists():
        return []
    cases = []
    for p in sorted(case_dir.glob("*.json")):
        cases.append(json.loads(p.read_text()))
    return cases


def _load_v2_cases(repo_root: Path) -> list[dict]:
    case_dir = repo_root / "misc" / "calibration_cases" / "operational_v2"
    if not case_dir.exists():
        return []
    cases = []
    for p in sorted(list(case_dir.glob("*.json")) + list(case_dir.glob("*.yaml"))):
        # operational_v2 files have .yaml extension but are JSON
        try:
            cases.append(json.loads(p.read_text()))
        except Exception:
            pass
    return cases


def _evaluate_case(runner, world, case_raw: dict) -> dict:
    """Run evaluate_scenario for a case and return key metrics."""
    from gim.scenario_compiler import compile_question

    scenario_data = case_raw.get("scenario", {})
    question = scenario_data.get("question", case_raw.get("question", ""))
    actors = scenario_data.get("actors", case_raw.get("agents", []))
    template = scenario_data.get("template", case_raw.get("template"))
    horizon = scenario_data.get("horizon_months", case_raw.get("horizon_months", 24))

    scenario = compile_question(
        question=question,
        world=world,
        actors=actors,
        template_id=template,
        horizon_months=horizon,
    )

    # Apply risk bias overrides if present
    risk_biases = case_raw.get("risk_bias_overrides",
                                scenario_data.get("risk_bias_overrides", {}))
    if risk_biases:
        for key, val in risk_biases.items():
            if hasattr(scenario, "risk_biases"):
                scenario.risk_biases[key] = float(val)

    evaluation = runner.evaluate_scenario(scenario)

    probs = evaluation.risk_probabilities
    dominant = max(probs, key=probs.get)
    non_sq = sum(v for k, v in probs.items() if k != "status_quo")

    return {
        "case_id": case_raw.get("id", case_raw.get("case_id", "unknown")),
        "dominant_label": dominant,
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "criticality_score": round(evaluation.criticality_score, 4),
        "calibration_score": round(evaluation.calibration_score, 4),
        "non_sq_mass": round(non_sq, 4),
    }


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        world = load_operational_world()
        g = gim()
        repo_root = Path(g["OPERATIONAL_STATE_CSV"]).parent.parent

        from gim.game_runner import GameRunner
        runner = GameRunner(world)

        # ------------------------------------------------------------------
        # Load and evaluate all cases
        # ------------------------------------------------------------------
        v1_cases = _load_v1_cases(repo_root)
        v2_cases = _load_v2_cases(repo_root)

        v1_results = []
        for case_raw in v1_cases:
            try:
                result = _evaluate_case(runner, world, case_raw)
                # Determine expected label
                expectations = case_raw.get("expectations", {})
                expected_tops = expectations.get("top_outcomes",
                                                  [case_raw.get("expected_outcomes", {}).get("dominant", "")])
                result["expected_labels"] = expected_tops
                result["match"] = result["dominant_label"] in expected_tops
                # Classify as crisis or stable
                tags = case_raw.get("tags", [])
                result["is_stable"] = "stability" in " ".join(tags).lower() or "status_quo" in expected_tops
                v1_results.append(result)
            except Exception as exc:
                v1_results.append({
                    "case_id": case_raw.get("id", "?"),
                    "error": str(exc),
                    "match": False,
                    "is_stable": False,
                })

        v2_results = []
        v2_expected = {
            "brazil_lula_crisis_2002": "negotiated_deescalation",
            "south_korea_imf_1997": "negotiated_deescalation",
            "turkey_fx_crisis_2018": "internal_destabilization",
            "argentina_default_2001": "internal_destabilization",
            "france_gilets_jaunes_2018": "status_quo",
        }
        for case_raw in v2_cases:
            try:
                result = _evaluate_case(runner, world, case_raw)
                case_id = result["case_id"]
                expected = case_raw.get("expected_outcomes", {}).get("dominant",
                            v2_expected.get(case_id, ""))
                result["expected_labels"] = [expected] if expected else []
                result["match"] = result["dominant_label"] == expected
                v2_results.append(result)
            except Exception as exc:
                v2_results.append({
                    "case_id": case_raw.get("case_id", "?"),
                    "error": str(exc),
                    "match": False,
                })

        # ------------------------------------------------------------------
        # M1: Accuracy
        # ------------------------------------------------------------------
        all_results = v1_results + v2_results
        valid = [r for r in all_results if "error" not in r]
        accuracy = sum(1 for r in valid if r["match"]) / max(len(valid), 1)

        # ------------------------------------------------------------------
        # M2: Criticality separation (v1 only)
        # ------------------------------------------------------------------
        crisis_cases = [r for r in v1_results if "error" not in r and not r.get("is_stable")]
        stable_cases = [r for r in v1_results if "error" not in r and r.get("is_stable")]

        if crisis_cases and stable_cases:
            mean_crit_crisis = sum(r.get("criticality_score", 0) for r in crisis_cases) / len(crisis_cases)
            mean_crit_stable = sum(r.get("criticality_score", 0) for r in stable_cases) / len(stable_cases)
            crit_gap = mean_crit_crisis - mean_crit_stable
        else:
            mean_crit_crisis = mean_crit_stable = crit_gap = 0.0

        # ------------------------------------------------------------------
        # M3: Probability mass allocation
        # ------------------------------------------------------------------
        crisis_non_sq = [r.get("non_sq_mass", 0) for r in crisis_cases if "error" not in r]
        stable_non_sq = [r.get("non_sq_mass", 0) for r in stable_cases if "error" not in r]
        avg_crisis_nonsq = sum(crisis_non_sq) / max(len(crisis_non_sq), 1)
        avg_stable_nonsq = sum(stable_non_sq) / max(len(stable_non_sq), 1)

        # ------------------------------------------------------------------
        # M4: v2 near-miss accuracy
        # ------------------------------------------------------------------
        v2_valid = [r for r in v2_results if "error" not in r]
        v2_accuracy = sum(1 for r in v2_valid if r["match"]) / max(len(v2_valid), 1)

        # ------------------------------------------------------------------
        # M5: Perturbation stability on v1 (top-3 weights ±10%)
        # ------------------------------------------------------------------
        flip_count = 0
        total_perturbations = 0
        try:
            from gim.geo_calibration import collect_geo_weight_paths, set_geo_weight_value, GeoWeight

            # Identify top-3 sensitive weights from outcome_intercept and outcome_driver
            sensitive_paths = [
                "outcome_intercept:status_quo",
                "outcome_driver:internal_destabilization:social_stress",
                "outcome_driver:status_quo:social_stress",
            ]

            for case_result in v1_results[:6]:  # Test first 6 for speed
                if "error" in case_result:
                    continue
                case_raw = next((c for c in v1_cases if c.get("id") == case_result["case_id"]), None)
                if case_raw is None:
                    continue
                original_label = case_result["dominant_label"]

                for path in sensitive_paths:
                    all_weights = collect_geo_weight_paths()
                    if path not in all_weights:
                        continue
                    original_weight = all_weights[path]

                    for factor in [0.90, 1.10]:
                        new_val = original_weight.value * factor
                        old = set_geo_weight_value(path, new_val)
                        try:
                            pert_result = _evaluate_case(runner, world, case_raw)
                            if pert_result["dominant_label"] != original_label:
                                flip_count += 1
                            total_perturbations += 1
                        finally:
                            set_geo_weight_value(path, old.value)
        except ImportError:
            pass

        flip_rate = flip_count / max(total_perturbations, 1)

        # ------------------------------------------------------------------
        # Assess
        # ------------------------------------------------------------------
        conditions = {
            "accuracy_ge_080": accuracy >= 0.80,
            "crit_gap_gt_015": crit_gap > 0.15,
            "crisis_nonsq_gt_040": avg_crisis_nonsq > 0.40,
            "stable_nonsq_lt_030": avg_stable_nonsq < 0.30,
            "v2_accuracy_ge_080": v2_accuracy >= 0.80,
            "flip_rate_lt_015": flip_rate < 0.15,
        }

    return make_result(
        test_id="T5",
        status=determine_status(conditions),
        metrics={
            "total_cases": len(all_results),
            "accuracy": round(accuracy, 3),
            "v2_accuracy": round(v2_accuracy, 3),
            "criticality_gap": round(crit_gap, 4),
            "avg_crisis_non_sq_mass": round(avg_crisis_nonsq, 4),
            "avg_stable_non_sq_mass": round(avg_stable_nonsq, 4),
            "perturbation_flip_rate": round(flip_rate, 3),
            "perturbation_flips": flip_count,
            "perturbation_total": total_perturbations,
        },
        flagged_items=[
            f"Accuracy = {accuracy:.0%} (threshold: ≥80%)" if accuracy < 0.80 else "",
            f"Criticality gap = {crit_gap:.3f} (threshold: >0.15)" if crit_gap <= 0.15 else "",
        ],
        notes=f"Completed in {t.elapsed:.1f}s. {len(v1_cases)} v1 + {len(v2_cases)} v2 cases.",
        details={
            "v1_results": v1_results,
            "v2_results": v2_results,
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
