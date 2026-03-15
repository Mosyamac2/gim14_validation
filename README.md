# GIM-14 Model Validation Suite

A comprehensive validation framework for the GIM-14 geopolitical simulation model,
implementing 12 qualitative architecture tests and 9 quantitative metric tests.

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

2. Install GIM-14 as an editable package and validation dependencies:
```bash
pip install -e /path/to/GIM-GIM14
pip install -r requirements.txt
```

3. For LLM tests (Q11, Q12, T6–T9), configure your API key:
```bash
cp .env.example .env
# Edit .env and set DEEPSEEK_API_KEY=sk-your-key-here
```

## Running Tests

```bash
# Run all 21 tests (requires DEEPSEEK_API_KEY for LLM tests)
python run_validation.py

# Run only non-LLM tests (no API key needed)
python run_validation.py --no-llm

# Run only LLM tests
python run_validation.py --llm

# Run specific tests
python run_validation.py Q1 Q5 T1 T6

# Run only qualitative or quantitative
python run_validation.py --qualitative
python run_validation.py --quantitative

# Custom output
python run_validation.py --output my_results.json
```

## Test Inventory

### Qualitative Tests (Q1–Q12)
| ID  | Name | Target | LLM? |
|-----|------|--------|------|
| Q1  | Scale-Factor Lock-In | economy.py `_scale_factor` freeze | No |
| Q2  | Damage Function Shape | climate.py damage multiplier curve | No |
| Q3  | LLM Prompt Homogeneity | policy.py prompt convergence risk | No |
| Q4  | Write Ordering Sensitivity | simulation.py multi-writer ordering | No |
| Q5  | Debt Crisis Haircut Paradox | social.py onset mechanics | No |
| Q6  | Softmax Outcome Structure | game_runner.py intercept bias | No |
| Q7  | Trade Recovery Absence | political_dynamics.py trade decay | No |
| Q8  | Credit Rating Feedback Lag | credit_rating → interest rate timing | No |
| Q9  | Actor Inference Coverage | scenario_compiler.py alias table | No |
| Q10 | Equilibrium Convergence | equilibrium_runner.py Hedge algorithm | No |
| Q11 | LLM Behavioral Heterogeneity | Full-trajectory archetype divergence | **Yes** |
| Q12 | LLM Fallback Rate | Silent degradation to simple policy | **Yes** |

### Quantitative Tests (T1–T9)
| ID | Name | Estimated Time | LLM? |
|----|------|---------------|------|
| T1 | Lyapunov Perturbation | ~60s | No |
| T2 | Parameter Sensitivity | ~10s | No |
| T3 | Crisis Replication | ~5s | No |
| T4 | Relation Audit | ~2s | No |
| T5 | Outcome Calibration | ~5s | No |
| T6 | LLM Sanity Bounds | ~2-3 min | **Yes** |
| T7 | LLM Reproducibility | ~4-5 min | **Yes** |
| T8 | LLM Policy Coherence | ~2-3 min | **Yes** |
| T9 | LLM vs Simple Divergence | ~2-3 min | **Yes** |

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
├── run_validation.py              # Main entry point / CLI
├── requirements.txt               # Python dependencies
├── .env.example                   # API key template — copy to .env
├── VALIDATION_METHODOLOGY.md      # Full methodology document
├── README.md
├── pyproject.toml
├── __init__.py
├── helpers.py                     # Shared utilities (world loading, LLM runs, patching)
├── qualitative/
│   ├── q01_scale_factor.py        # Q1:  Production function scale-factor lock-in
│   ├── q02_damage_function.py     # Q2:  Climate damage function shape
│   ├── q03_llm_homogeneity.py     # Q3:  LLM prompt convergence analysis
│   ├── q04_write_ordering.py      # Q4:  Multi-writer state ordering sensitivity
│   ├── q05_debt_haircut.py        # Q5:  Debt crisis onset haircut paradox
│   ├── q06_softmax_structure.py   # Q6:  Softmax outcome model bias
│   ├── q07_trade_recovery.py      # Q7:  Trade intensity irreversibility
│   ├── q08_credit_lag.py          # Q8:  Credit rating → interest rate lag
│   ├── q09_actor_inference.py     # Q9:  Scenario compiler actor resolution
│   ├── q10_equilibrium.py         # Q10: Hedge equilibrium convergence
│   ├── q11_llm_heterogeneity.py   # Q11: LLM behavioral heterogeneity [LLM]
│   └── q12_llm_fallback.py        # Q12: LLM fallback rate [LLM]
└── quantitative/
    ├── t01_lyapunov.py            # T1:  Initial-condition perturbation
    ├── t02_sensitivity.py         # T2:  Parameter sensitivity sweep
    ├── t03_crisis_replication.py   # T3:  Argentina/Turkey crisis replication
    ├── t04_relation_audit.py      # T4:  Bilateral relation asymmetry audit
    ├── t05_outcome_calibration.py # T5:  Outcome probability calibration
    ├── t06_llm_sanity.py          # T6:  LLM trajectory sanity bounds [LLM]
    ├── t07_llm_reproducibility.py # T7:  LLM run reproducibility [LLM]
    ├── t08_llm_coherence.py       # T8:  LLM policy coherence rules [LLM]
    └── t09_llm_vs_simple.py       # T9:  LLM vs simple divergence [LLM]
```
