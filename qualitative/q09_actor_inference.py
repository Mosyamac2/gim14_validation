"""
Q9 — Scenario Compiler: Deterministic Actor Inference Without Geopolitical Context

Test whether the scenario compiler correctly resolves actors from
geopolitically meaningful questions, or silently falls back to GDP-top-3.
"""
from __future__ import annotations
from ..helpers import gim, load_operational_world, make_result, determine_status, Timer


TEST_QUESTIONS = [
    {
        "question": "What if BRICS abandons the dollar?",
        "expected_actors": ["Brazil", "Russia", "India", "China", "South Africa"],
        "min_expected_resolved": 3,
    },
    {
        "question": "How would a NATO-Russia confrontation over the Baltics unfold?",
        "expected_actors": ["United States", "Russia", "Germany", "France", "United Kingdom"],
        "min_expected_resolved": 2,
    },
    {
        "question": "What happens if there is a drought crisis in Sub-Saharan Africa?",
        "expected_actors": ["Nigeria", "South Africa", "Kenya", "Ethiopia"],
        "min_expected_resolved": 1,
    },
    {
        "question": "What if North Korea tests a nuclear device?",
        "expected_actors": ["South Korea", "Japan", "China", "United States"],
        "min_expected_resolved": 2,
    },
    {
        "question": "How would OPEC production cuts affect global stability?",
        "expected_actors": ["Saudi Arabia", "Russia", "United States", "Iran"],
        "min_expected_resolved": 1,
    },
]


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        world = load_operational_world()
        from gim.scenario_compiler import (
            compile_question, infer_actor_names, resolve_actor_names,
        )

        results = []
        total_correct = 0
        total_questions = len(TEST_QUESTIONS)

        for tc in TEST_QUESTIONS:
            inferred_names = infer_actor_names(tc["question"], world)
            actor_ids, actor_names, unresolved = resolve_actor_names(world, inferred_names)

            # Count how many expected actors were found
            resolved_set = set(actor_names)
            expected_set = set(tc["expected_actors"])
            matched = resolved_set & expected_set
            match_count = len(matched)

            # Check if this is a fallback (top-3 by GDP)
            top3 = sorted(world.agents.values(), key=lambda a: a.economy.gdp, reverse=True)[:3]
            top3_names = {a.name for a in top3}
            is_fallback = resolved_set == top3_names and not (resolved_set & expected_set)

            passed = match_count >= tc["min_expected_resolved"]
            if passed:
                total_correct += 1

            results.append({
                "question": tc["question"],
                "expected_actors": tc["expected_actors"],
                "resolved_actors": actor_names,
                "unresolved": unresolved,
                "matched_count": match_count,
                "min_expected": tc["min_expected_resolved"],
                "is_fallback": is_fallback,
                "passed": passed,
            })

        accuracy = total_correct / total_questions if total_questions > 0 else 0
        fallback_count = sum(1 for r in results if r["is_fallback"])

        conditions = {
            "accuracy_ge_60pct": accuracy >= 0.60,
            "no_silent_fallback": fallback_count == 0,
        }

    return make_result(
        test_id="Q9",
        status=determine_status(conditions),
        metrics={
            "accuracy": round(accuracy, 2),
            "questions_passed": total_correct,
            "total_questions": total_questions,
            "silent_fallback_count": fallback_count,
        },
        flagged_items=[
            f"Question failed: '{r['question']}' → resolved {r['resolved_actors']}"
            for r in results if not r["passed"]
        ],
        notes=f"Completed in {t.elapsed:.1f}s",
        details={
            "results": results,
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
