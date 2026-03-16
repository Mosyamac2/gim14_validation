#!/usr/bin/env python3
"""
GIM-14 Validation Suite Runner

Usage:
    python run_validation.py                         # Run all tests
    python run_validation.py Q1 Q5 T1 T3             # Run specific tests
    python run_validation.py --qualitative            # Run only qualitative
    python run_validation.py --quantitative           # Run only quantitative
    python run_validation.py --output results.json    # Custom output path
    python run_validation.py --verbose                # Verbose mode (enables optional LLM calls)

Environment:
    GIM14_REPO   Path to the GIM-14 repository root (auto-detected if adjacent)
    DEEPSEEK_API_KEY   Optional, enables LLM calls in Q3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent      # .../GIM-14/gim14_validation/
PARENT_OF_PROJECT = PROJECT_ROOT.parent             # .../GIM-14/
sys.path.insert(0, str(PARENT_OF_PROJECT))          # now `import gim14_validation` works


# ---------------------------------------------------------------------------
# Test registry: maps test IDs to their module paths
# ---------------------------------------------------------------------------
TEST_REGISTRY = {
    # Qualitative
    "Q1":  "gim14_validation.qualitative.q01_scale_factor",
    "Q2":  "gim14_validation.qualitative.q02_damage_function",
    "Q3":  "gim14_validation.qualitative.q03_llm_homogeneity",
    "Q4":  "gim14_validation.qualitative.q04_write_ordering",
    "Q5":  "gim14_validation.qualitative.q05_debt_haircut",
    "Q6":  "gim14_validation.qualitative.q06_softmax_structure",
    "Q7":  "gim14_validation.qualitative.q07_trade_recovery",
    "Q8":  "gim14_validation.qualitative.q08_credit_lag",
    "Q9":  "gim14_validation.qualitative.q09_actor_inference",
    "Q10": "gim14_validation.qualitative.q10_equilibrium",
    "Q11": "gim14_validation.qualitative.q11_llm_heterogeneity",
    "Q12": "gim14_validation.qualitative.q12_llm_fallback",
    # Quantitative
    "T1":  "gim14_validation.quantitative.t01_lyapunov",
    "T2":  "gim14_validation.quantitative.t02_sensitivity",
    "T3":  "gim14_validation.quantitative.t03_crisis_replication",
    "T4":  "gim14_validation.quantitative.t04_relation_audit",
    "T5":  "gim14_validation.quantitative.t05_outcome_calibration",
    "T6":  "gim14_validation.quantitative.t06_llm_sanity",
    "T7":  "gim14_validation.quantitative.t07_llm_reproducibility",
    "T8":  "gim14_validation.quantitative.t08_llm_coherence",
    "T9":  "gim14_validation.quantitative.t09_llm_vs_simple",
}

QUALITATIVE_IDS = [k for k in TEST_REGISTRY if k.startswith("Q")]
QUANTITATIVE_IDS = [k for k in TEST_REGISTRY if k.startswith("T")]
LLM_TEST_IDS = ["Q11", "Q12", "T6", "T7", "T8", "T9"]
NON_LLM_IDS = [k for k in TEST_REGISTRY if k not in LLM_TEST_IDS]


def import_and_run(test_id: str, verbose: bool = False) -> dict:
    """Import a test module and run it, catching all exceptions."""
    module_path = TEST_REGISTRY.get(test_id)
    if module_path is None:
        return {
            "test_id": test_id,
            "status": "ERROR",
            "metrics": {},
            "notes": f"Unknown test ID: {test_id}",
        }
    try:
        import importlib
        module = importlib.import_module(module_path)
        return module.run(verbose=verbose)
    except Exception as exc:
        return {
            "test_id": test_id,
            "status": "ERROR",
            "metrics": {},
            "notes": f"Exception: {exc}",
            "traceback": traceback.format_exc(),
        }


def run_suite(test_ids: list[str], verbose: bool = False) -> dict:
    """Run a list of tests and assemble the combined result."""
    results = []
    overall_start = time.perf_counter()

    for test_id in test_ids:
        print(f"  Running {test_id}...", end=" ", flush=True)
        start = time.perf_counter()
        result = import_and_run(test_id, verbose=verbose)
        elapsed = time.perf_counter() - start
        status = result.get("status", "?")
        symbol = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "ERROR": "☠"}.get(status, "?")
        print(f"{symbol} {status} ({elapsed:.1f}s)")
        results.append(result)

    total_elapsed = time.perf_counter() - overall_start

    # Summarise
    pass_count = sum(1 for r in results if r.get("status") == "PASS")
    warn_count = sum(1 for r in results if r.get("status") == "WARN")
    fail_count = sum(1 for r in results if r.get("status") == "FAIL")
    error_count = sum(1 for r in results if r.get("status") == "ERROR")

    return {
        "suite": "GIM-14 Validation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tests": len(results),
        "passed": pass_count,
        "warned": warn_count,
        "failed": fail_count,
        "errors": error_count,
        "elapsed_seconds": round(total_elapsed, 1),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="GIM-14 Model Validation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "tests", nargs="*", default=[],
        help="Test IDs to run (e.g. Q1 Q5 T1). Default: all.",
    )
    parser.add_argument(
        "--qualitative", action="store_true",
        help="Run only qualitative tests (Q1-Q10).",
    )
    parser.add_argument(
        "--quantitative", action="store_true",
        help="Run only quantitative tests (T1-T9).",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Run only LLM-dependent tests (Q11, Q12, T6-T9). Requires DEEPSEEK_API_KEY.",
    )
    parser.add_argument(
        "--no-llm", action="store_true", dest="no_llm",
        help="Run only non-LLM tests (Q1-Q10, T1-T5).",
    )
    parser.add_argument(
        "--output", "-o", default="validation_results.json",
        help="Output JSON file path (default: validation_results.json).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose mode: enable optional LLM API calls.",
    )
    args = parser.parse_args()

    # Resolve test list
    if args.tests:
        test_ids = [tid.upper() for tid in args.tests]
        unknown = [tid for tid in test_ids if tid not in TEST_REGISTRY]
        if unknown:
            print(f"Unknown test IDs: {unknown}")
            print(f"Available: {sorted(TEST_REGISTRY.keys())}")
            sys.exit(1)
    elif args.qualitative:
        test_ids = QUALITATIVE_IDS
    elif args.quantitative:
        test_ids = QUANTITATIVE_IDS
    elif args.llm:
        test_ids = LLM_TEST_IDS
    elif args.no_llm:
        test_ids = NON_LLM_IDS
    else:
        test_ids = sorted(TEST_REGISTRY.keys())

    print("=" * 60)
    print("  GIM-14 Model Validation Suite")
    print(f"  Tests: {', '.join(test_ids)}")
    print("=" * 60)

    suite_result = run_suite(test_ids, verbose=args.verbose)

    # Print summary
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total:   {suite_result['total_tests']}")
    print(f"  Passed:  {suite_result['passed']}")
    print(f"  Warned:  {suite_result['warned']}")
    print(f"  Failed:  {suite_result['failed']}")
    print(f"  Errors:  {suite_result['errors']}")
    print(f"  Time:    {suite_result['elapsed_seconds']}s")
    print("=" * 60)

    # Write JSON output
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(suite_result, f, indent=2, default=str)
    print(f"\n  Results written to: {output_path}")


if __name__ == "__main__":
    main()
