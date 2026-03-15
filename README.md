# GIM-14 Model Validation Suite

A comprehensive validation framework for the GIM-14 geopolitical simulation model,
implementing 10 qualitative architecture tests and 5 quantitative metric tests.

## Setup

1. Place this project **adjacent** to the GIM-14 repository, or set `GIM14_REPO`:

```
parent_directory/
├── GIM-GIM14/          # or GIM_14/
│   ├── gim/
│   ├── data/
│   └── ...
└── gim14_validation/   # this project
    ├── run_validation.py
    └── ...
```

Alternatively:
```bash
export GIM14_REPO=/path/to/GIM_14
```

2. Install GIM-14 dependencies (Python 3.10+):
```bash
pip install -e /path/to/GIM_14
```

3. Optional: set `DEEPSEEK_API_KEY` for LLM tests in Q3.

## Running Tests

```bash
# Run all 15 tests
python run_validation.py

# Run specific tests
python run_validation.py Q1 Q5 T1 T3

# Run only qualitative or quantitative
python run_validation.py --qualitative
python run_validation.py --quantitative

# Verbose mode (enables optional LLM calls)
python run_validation.py --verbose

# Custom output
python run_validation.py --output my_results.json
```

## Test Inventory

### Qualitative Tests (Q1–Q10)
| ID  | Name | Target |
|-----|------|--------|
| Q1  | Scale-Factor Lock-In | economy.py `_scale_factor` freeze |
| Q2  | Damage Function Shape | climate.py damage multiplier curve |
| Q3  | LLM Prompt Homogeneity | policy.py prompt convergence risk |
| Q4  | Write Ordering Sensitivity | simulation.py multi-writer ordering |
| Q5  | Debt Crisis Haircut Paradox | social.py onset mechanics |
| Q6  | Softmax Outcome Structure | game_runner.py intercept bias |
| Q7  | Trade Recovery Absence | political_dynamics.py trade decay |
| Q8  | Credit Rating Feedback Lag | credit_rating → interest rate timing |
| Q9  | Actor Inference Coverage | scenario_compiler.py alias table |
| Q10 | Equilibrium Convergence | equilibrium_runner.py Hedge algorithm |

### Quantitative Tests (T1–T5)
| ID | Name | Estimated Time |
|----|------|---------------|
| T1 | Lyapunov Perturbation | ~60s |
| T2 | Parameter Sensitivity | ~10s |
| T3 | Crisis Replication | ~5s |
| T4 | Relation Audit | ~2s |
| T5 | Outcome Calibration | ~5s |

## Output

Results are saved as structured JSON:

```json
{
  "suite": "GIM-14 Validation",
  "timestamp": "2026-03-16T...",
  "total_tests": 15,
  "passed": 10,
  "warned": 3,
  "failed": 2,
  "errors": 0,
  "results": [
    {
      "test_id": "Q1",
      "status": "PASS",
      "metrics": { ... },
      "flagged_items": [],
      "notes": "..."
    }
  ]
}
```

## Project Structure

```
gim14_validation/
├── run_validation.py          # Main entry point / CLI
├── __init__.py
├── helpers.py                 # Shared utilities
├── qualitative/
│   ├── q01_scale_factor.py
│   ├── q02_damage_function.py
│   ├── q03_llm_homogeneity.py
│   ├── q04_write_ordering.py
│   ├── q05_debt_haircut.py
│   ├── q06_softmax_structure.py
│   ├── q07_trade_recovery.py
│   ├── q08_credit_lag.py
│   ├── q09_actor_inference.py
│   └── q10_equilibrium.py
├── quantitative/
│   ├── t01_lyapunov.py
│   ├── t02_sensitivity.py
│   ├── t03_crisis_replication.py
│   ├── t04_relation_audit.py
│   └── t05_outcome_calibration.py
├── pyproject.toml
└── README.md
```
