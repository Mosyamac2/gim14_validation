# GIM-14 Model Validation Methodology

**Version:** 1.0  
**Date:** 2026-03-16  
**Scope:** Systematic validation of the GIM-14 geopolitical simulation model covering qualitative architecture review and quantitative stress testing.

---

## 1. Preamble

GIM-14 is a multi-agent yearly simulation model that takes a natural-language political scenario, compiles it into a structured geopolitical game, and produces probabilistic outcome distributions over a 3-year horizon. Agents represent sovereign states (up to 57 actors) and interact through economic, resource, climate, social, political, and security channels. The model has two conceptual layers: (i) a deterministic yearly world-step engine with endogenous feedback loops, and (ii) a scenario evaluation and game-theory overlay that scores outcome probabilities and player payoffs.

This document defines a validation methodology consisting of 10 qualitative tests (architecture and theory review) and 5 quantitative tests (numerical sensitivity and metric checks). Each test is designed to be specific to GIM-14 code and architecture, actionable by a coder, and computationally lightweight (single-seed short runs, no LLM API calls unless explicitly noted).

---

## Part A — Qualitative Tests

Each qualitative test targets a specific architectural or theoretical vulnerability in GIM-14. For every test we state: what to inspect, why it matters, how to carry it out, and what constitutes a pass/fail with a concrete recommendation.

---

### Q1. Production Function Energy Exponent and Scale-Factor Lock-In

**What to inspect.** In `economy.py`, GDP is computed via a Cobb-Douglas-with-energy production function: `GDP = TFP · tech_factor · K^α · L^β · E^γ`. The exponents are `α=0.30, β=0.60, γ=0.07`, summing to `0.97` — slightly decreasing returns to scale. On the first call, a `_scale_factor` is lazily computed as `GDP_initial / GDP_potential` and then permanently frozen for the rest of the simulation.

**Why it matters.** The frozen `_scale_factor` absorbs all calibration residuals at t=0 into a single country-specific constant. This means any initial-state data error (e.g., incorrect energy consumption, wrong capital estimate from the `3×GDP` heuristic) is permanently baked into the entire trajectory as a level shift. Over a 10-year run, a country whose capital was mis-initialized by 30% will carry that exact bias through every year, because the scale factor never re-adapts. This creates a hidden coupling between data quality and GDP trajectory realism that is not visible in the calibration surface.

**How to carry out.**

1. Load the 57-actor operational state. Run 1 step to trigger `_scale_factor` initialization.
2. Record the distribution of `_scale_factor` across all agents. Flag agents where `_scale_factor > 2.0` or `_scale_factor < 0.3` — these indicate that the production function structure is a poor fit for the initial data.
3. For the 5 agents with the most extreme scale factors, perturb `capital` by ±20% in the initial CSV and re-run 5 steps. Measure the resulting GDP trajectory shift as a fraction of baseline GDP.
4. Assess whether the scale-factor lock-in transmits initial-state errors linearly or amplifies them.

**Pass criterion.** The distribution of `_scale_factor` should be unimodal with coefficient of variation < 1.0. GDP trajectory shift from ±20% capital perturbation should be bounded within ±15% of baseline GDP at t+5 for at least 80% of tested agents.

**Recommendation if fail.** Introduce a slow-adapting scale factor (e.g., exponential smoothing with τ=5 years) so that initial-state errors decay rather than persist indefinitely. Alternatively, add a calibration diagnostic that flags agents with extreme scale factors before simulation begins.

---

### Q2. Climate Damage Function Shape at Model-Relevant Temperatures

**What to inspect.** In `climate.py`, the `climate_damage_multiplier` function has the form: `multiplier = 1 + benefit_gaussian(ΔT) − 0.006·T²`, where `T` is the absolute global temperature (not the anomaly). The benefit Gaussian peaks at `ΔT=0.30°C` above the 2023 baseline (`~1.29°C` anomaly). The quadratic term uses absolute temperature, not anomaly.

**Why it matters.** Two problems: (a) The quadratic damage uses absolute global temperature (roughly 1.3–2.5°C in anomaly terms), not the temperature anomaly. This means the damage baseline at `T=1.29°C` is already `0.006 × 1.29² ≈ 0.010`, giving a ~1% GDP penalty just from the current temperature. For a 2°C anomaly, damage is `0.006 × 4 = 0.024` — only a 2.4% GDP loss, which is at the low end of DICE and far below integrated assessment models like PAGE or FUND for warming above 2°C. (b) The benefit Gaussian (peak +0.6% GDP at +0.3°C warming) creates a nonmonotonic damage function where slight warming is economically beneficial. While this follows Nordhaus (2017), it is contested and may produce the counterintuitive result that moderate warming scenarios score better economically than the baseline.

**How to carry out.**

1. Evaluate `climate_damage_multiplier(T)` for `T` in `[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]` (these are anomaly values that the model can reach over 10–30 year horizons).
2. Plot the multiplier curve and identify the peak-benefit temperature and the zero-crossing temperature (where damages exactly offset benefits).
3. Compare the model's damage at +2°C and +3°C anomaly against DICE-2016R2, PAGE09, and FUND reference values.
4. Run a 10-step simulation with and without the benefit Gaussian (set `DAMAGE_BENEFIT_MAX=0`) and compare GDP and climate-policy behavior trajectories.

**Pass criterion.** The damage function at +3°C anomaly should impose at least 2% GDP loss (literature central estimate is 2–5%). The benefit Gaussian should not cause the model to prefer a warming path over the baseline in any 10-step run.

**Recommendation if fail.** Replace absolute temperature in the quadratic with the anomaly (i.e., `T - T_preindustrial`), and recalibrate `DAMAGE_QUAD_COEFF` against the IPCC AR6 WGII damage corridor. Consider removing or substantially shrinking the benefit Gaussian unless it can be defended with post-AR6 literature. Both parameters are currently tagged `[PRIOR]`.

---

### Q3. LLM Policy Prompt — Structural Homogeneity and Behavioral Convergence Risk

**What to inspect.** In `policy.py`, the `LLM_POLICY_PROMPT_TEMPLATE` is a single fixed prompt sent to DeepSeek for all 57 agents. The prompt contains behavioral-heterogeneity instructions (use your PDI, IDV, MAS, regime_type, alliance_block) and explicit escalation ladders. The observation JSON is agent-specific, but the system instructions, win/loss conditions, and heuristics are identical.

**Why it matters.** Despite the heterogeneity instructions, the LLM receives the same meta-objective (grow GDP, maintain stability, keep security_margin ≥ 1.0) for every agent. This creates a structural convergence risk: a cost-minimizing LLM will tend to find a single policy archetype that satisfies the prompt constraints regardless of cultural or regime inputs. Specifically: (a) The climate-policy section explicitly tells agents with low GDP per capita to choose weak/no climate policy — this is a reasonable heuristic but it hardcodes a specific political-economy theory into the prompt rather than letting it emerge. (b) The escalation ladder gives exact thresholds (conflict_level > 0.60 for border_incident), which means all agents will use the same escalation trigger regardless of their risk tolerance or strategic culture. (c) The prompt is ~3500 tokens of instructions plus ~4–9KB of observation JSON. With DeepSeek-chat at temperature 0.2, responses will cluster tightly around the prompt's modal interpretation.

**How to carry out.**

1. Without calling the LLM, construct 6 synthetic observation payloads representing archetypal countries: (a) wealthy democracy high-trust, (b) wealthy autocracy low-trust, (c) poor fragile state, (d) resource-rich MENA state, (e) rising power high-growth, (f) small open economy. Give each the same neighbor set but vary the agent-specific state.
2. For each archetype, manually predict what the prompt would elicit from a language model by tracing the explicit heuristics and thresholds. Count how many of the 6 archetypes would produce distinguishable policy vectors (different climate_policy, different security_actions, different trade_deal patterns).
3. If an LLM API is available: send all 6 prompts with temperature=0.2 and record the action JSONs. Compute the pairwise Jaccard distance on the categorical fields (climate_policy, security_actions.type) and the L1 distance on the continuous fields (spending changes). A healthy model should show inter-archetype distance > intra-archetype distance.

**Pass criterion.** At least 4 of the 6 archetypes should produce qualitatively distinct policy vectors (different climate_policy level, different security action type, or spending-change signs that differ). If all 6 converge to the same action profile, the heterogeneity instructions are ineffective.

**Recommendation if fail.** Split the prompt into archetype-specific prompt variants indexed by regime_type × income_group × alliance_block (a 3×3×3 = 27-cell grid collapsible to ~6 templates). Each template should have different win conditions, risk tolerances, and escalation norms. Remove hardcoded threshold numbers from the prompt and replace with relative language ("when conflict is unusually high for your region").

---

### Q4. Multi-Writer State Mutation Ordering Sensitivity

**What to inspect.** The AUDIT_GIM14 document (and `SIMULATION_STEP_ORDER.md`) confirms that `economy.gdp`, `economy.capital`, `society.trust_gov`, and `society.social_tension` are written by 4–5 different modules within a single yearly step (actions, economy, geopolitics, climate, social, institutions). The writes are applied sequentially in the order dictated by `simulation.py`, and each writer reads the current (already-mutated) value, not the beginning-of-step snapshot.

**Why it matters.** Because each writer modifies the state variable in-place and the next writer reads the mutated value, the results depend on the ordering of the 30 sub-steps. For example, if `geopolitics.py` reduces GDP by 20% (war damage) before `economy.py` runs, the economy module sees a lower GDP base and produces a different capital investment. If the order were reversed, the economy would invest based on pre-war GDP, and the war damage would be applied to the post-investment GDP. This is not necessarily wrong — sequential ordering is a valid modeling choice — but it means that the specific ordering chosen is itself a model assumption that is not calibrated or validated.

**How to carry out.**

1. Identify the 4 most impactful write sites for `economy.gdp`: (a) `actions.py` (domestic policy effects), (b) `geopolitics.py` (war/sanctions GDP hit), (c) `economy.py` (production function update), (d) `social.py` (crisis onset GDP multiplier).
2. Create a single-country test harness with an agent in a stressed state (debt/GDP=1.0, conflict_level=0.6, trust=0.3, tension=0.6) where multiple writers will fire on the same step.
3. Run 3 steps with the current ordering.
4. Then swap the order of the economy and geopolitics sub-steps (move `update_economy_output` before `apply_security_actions`) and re-run.
5. Measure the GDP, trust, and tension divergence at t+3.

**Pass criterion.** GDP divergence between the two orderings at t+3 should be < 5% for a moderately stressed agent. If divergence > 10%, the ordering is load-bearing and must be treated as a calibrated structural assumption.

**Recommendation if fail.** Document the current ordering as a deliberate calibration choice with the same provenance discipline as numerical parameters. Consider refactoring the writes to operate on beginning-of-step snapshots with end-of-step batch commits, so that ordering within a sub-step group does not matter.

---

### Q5. Debt Crisis Onset Haircut Paradox

**What to inspect.** In `social.py:check_debt_crisis()`, the onset year applies `debt *= 0.60` and `gdp *= 0.90`. This means a country that enters debt crisis at `debt/GDP = 1.25` immediately moves to `debt/GDP = (1.25 × 0.60) / 0.90 = 0.833`, which is below the exit threshold of `0.70 × ... = 0.70`. The AUDIT_GIM14 confirms: "crisis started on step 1, but onset moved debt/GDP to 0.7585, so the debt crisis cleared on step 2."

**Why it matters.** The debt-crisis mechanism is designed to model prolonged fiscal distress (Argentina 2001, Greece 2010), but the onset haircut is so large relative to the GDP hit that it mechanically resolves the crisis trigger within 1–2 steps for any country near the entry threshold. This means the persistence mechanism (calibrated against Argentina and South Korea) is rarely activated for realistic initial conditions. The model effectively has a one-shot debt crisis rather than the intended multi-year episode. This is a known issue per the AUDIT but has not been resolved.

**How to carry out.**

1. Initialize a single agent at `debt/GDP` values of `[1.25, 1.50, 1.80, 2.00, 2.50, 3.00]` with `interest_rate > 0.12` (to trigger the crisis).
2. Run 6 steps for each initial condition with `POLICY_MODE=simple` and `DISABLE_EXTREME_EVENTS=1`.
3. Record the `debt_crisis_active_years` at each step. Identify the minimum initial `debt/GDP` at which the crisis persists for ≥ 3 years.
4. Compare against the real-world stylized fact that sovereign debt crises typically last 5–8 years (Reinhart & Rogoff, 2009).

**Pass criterion.** Debt crisis should persist for ≥ 3 years when the initial `debt/GDP` is between 1.2 and 1.5 (the range that historically triggers restructuring). Currently, the model clears the crisis within 1 step at `debt/GDP = 1.25`.

**Recommendation if fail.** Decouple the onset haircut from the exit condition. Options: (a) reduce `DEBT_CRISIS_DEBT_MULT` from 0.60 to 0.85 so the ratio doesn't drop below exit threshold on impact; (b) impose a minimum crisis duration of 2 years before exit is evaluated (partially implemented: `recovery_window_open = active_years >= 2`, but the onset haircut already resolves the ratio); (c) make the exit threshold depend on both the ratio and the interest rate simultaneously.

---

### Q6. Scenario Outcome Model — Softmax Over Linear Scores with Expert Priors

**What to inspect.** In `game_runner.py`, outcome probabilities are computed by: (1) building linear scores from `OUTCOME_INTERCEPTS + Σ(driver × weight) + action_shifts + shock_shifts + link_shifts + tail_risk`, then (2) applying softmax. All ~40 weights in the outcome layer are tagged `[PRIOR]` (expert priors) with CI95 half-widths of 0.04–0.10.

**Why it matters.** The softmax-over-linear-scores architecture has two structural issues: (a) **Softmax sensitivity to intercept scale.** The `status_quo` intercept is 1.20, while `broad_regional_escalation` is −0.35. This 1.55-unit gap means that even before any driver activates, the softmax will assign roughly `e^1.55 ≈ 4.7×` more probability to status_quo than to broad_regional_escalation. The driver weights (0.15–0.95) can only partially overcome this gap. This design choice implies that the model is structurally anchored to status_quo, which may be realistic but makes it difficult for stress scenarios to shift the distribution into crisis outcomes without stacking multiple drivers. (b) **No interaction terms.** The linear additive structure cannot capture the well-documented nonlinear interaction between conflict stress and resource stress (resource wars), or between debt stress and political instability (the fiscal-political doom loop). Two moderate-level stresses will always score less than one high-level stress, which contradicts the political science literature on compound crises.

**How to carry out.**

1. Compute the baseline softmax distribution with all drivers set to zero (only intercepts active). Record the probability mass on status_quo versus the sum of all crisis outcomes.
2. Set all drivers to their 75th percentile stress values simultaneously and recompute. Measure how much the crisis mass increases.
3. Compare with a counterfactual where a single driver (e.g., conflict_stress) is set to its 99th percentile alone. If the single-extreme-driver scenario produces more crisis mass than the compound scenario, the lack of interaction terms is empirically distorting.

**Pass criterion.** The compound stress scenario (all drivers at 75th percentile) should produce at least as much crisis probability mass as any single-driver-at-99th-percentile scenario. The baseline status_quo mass should be < 60% (otherwise the model is structurally incapable of predicting crises from moderate compound stress).

**Recommendation if fail.** Add pairwise interaction terms for the three most important compound channels: `conflict_stress × resource_gap`, `debt_stress × social_stress`, and `sanctions_pressure × energy_dependence`. Alternatively, replace the linear score model with a shallow neural network or a decision-tree scoring function calibrated against the operational_v2 near-miss cases.

---

### Q7. Endogenous Relation Drift — Absence of Trade Recovery Mechanism

**What to inspect.** In `political_dynamics.py:update_relations_endogenous()`, trust and conflict evolve through incremental drift terms. Trade intensity decays through `apply_trade_barrier_effects()` as: `trade_intensity *= (1 − 0.05 × barrier − friction)`. There is a trust mean-reversion toward 0.6 and a conflict mean-reversion toward 0.1, but there is no explicit recovery mechanism for trade_intensity.

**Why it matters.** Trade intensity can only decay (through barriers, conflict, and tension) but has no endogenous channel to recover after barriers are lifted. The `apply_trade_deals()` function in `actions.py` increases trade volume for specific deals, but it does not restore the underlying `trade_intensity` in the relation graph. Over a 10-year run, even transient trade barriers will permanently reduce trade intensity, creating an irreversible deglobalization drift. This is unrealistic: historical episodes (e.g., US-China after 2019, EU-Russia partial substitution) show that trade partially recovers when political conditions improve.

**How to carry out.**

1. Initialize a 2-country world with `trade_intensity=0.5, trade_barrier=0.0, trust=0.6, conflict=0.1`.
2. At t=1, inject a `trade_barrier=0.4` shock. At t=3, remove it (set barrier back to 0.0).
3. Run 10 steps total with simple policy. Record the `trade_intensity` trajectory.
4. Verify whether trade_intensity recovers toward its pre-shock level after the barrier is removed, or permanently settles at a lower level.

**Pass criterion.** After barrier removal, trade_intensity should recover to at least 80% of its pre-shock level within 5 years. If it does not, the model has a permanent-scarring assumption that is not documented.

**Recommendation if fail.** Add a trade-recovery drift term in `update_relations_endogenous()`: when `trade_barrier < 0.1` and `trust > 0.4`, apply `trade_intensity += recovery_rate × (baseline_trade − trade_intensity)` with `recovery_rate ≈ 0.05`. This mirrors the existing trust and conflict mean-reversion logic.

---

### Q8. Credit Rating → Interest Rate Feedback Gap

**What to inspect.** The AUDIT_GIM14 (item 2f) notes that `credit_rating.py` computes a detailed sovereign risk score but that `economy.py:compute_effective_interest_rate()` does not read `credit_rating` or `credit_zone`. Examination of `economy.py` shows that `_credit_zone_premium()` IS called and adds a zone-based premium, reading `agent.credit_zone`. However, the credit rating is updated at step 29 (end of year), while `update_public_finances` runs at step 21. This means the credit zone used for interest-rate computation is always from the **previous year**, not the current year.

**Why it matters.** The one-year lag in the credit-rating → interest-rate channel weakens the debt-crisis feedback loop. A country that enters distress in year T will only see its interest rate increase in year T+1, allowing it an extra year of cheap borrowing. For fast-moving crises (Argentine-style sudden stops), this lag underestimates the speed of the debt spiral. More importantly, this timing issue means that the credit-zone premium observed in the first simulation step is always the initialized value, not an endogenous one.

**How to carry out.**

1. Create an agent with `debt/GDP=1.0` and initial `credit_zone="investment"`.
2. Run 5 steps. Record the effective interest rate and the credit_zone at each step.
3. Verify whether the interest-rate increase from credit deterioration appears with a 0-year or 1-year lag.
4. Compare debt trajectories between the current (lagged) implementation and a patched version where `update_credit_ratings` is called before `update_public_finances`.

**Pass criterion.** This is a documentation/design test. The lag should be explicitly documented as intentional (conservative design) or flagged as a timing bug. If the debt trajectory diverges by more than 5% at t+5 between lagged and non-lagged implementations, the effect is material and should be addressed.

**Recommendation if fail.** Either move `update_credit_ratings` before `update_public_finances` in the yearly step order, or add a mid-year provisional credit assessment that feeds the interest-rate computation. Document the chosen timing as a calibrated design decision.

---

### Q9. Scenario Compiler — Deterministic Actor Inference Without Geopolitical Context

**What to inspect.** In `scenario_compiler.py:infer_actor_names()`, actors are extracted from a natural-language question using string matching against country names and a hardcoded alias table of ~12 entries. If no actors are found, the top 3 by GDP are used as a fallback. The scenario template is selected by keyword detection in `scenario_library.py:detect_template()`.

**Why it matters.** (a) **Alias coverage.** The alias table covers major powers (US, China, Japan, Germany, Saudi Arabia, Turkey, Iran, Israel) but misses common geopolitical references like "EU," "NATO," "OPEC," "BRICS," "North Korea," "DPRK," "South Africa," "UAE," "Egypt," or "Pakistan." A question like "What happens if BRICS imposes a counter-sanctions regime?" will fail to resolve any actors and default to top-3 by GDP (US, China, Japan) — which is exactly wrong for the scenario. (b) **No indirect actor inference.** The compiler cannot identify implied actors. "What happens if there is a war in the South China Sea?" should involve China, Taiwan, USA, and Philippines, but only "China" will be matched by the alias table. (c) **Fallback to GDP-top-3.** This fallback silently produces a plausible-looking but wrong actor set. The user will receive a well-formatted scenario evaluation for the wrong countries with no warning.

**How to carry out.**

1. Compile the following 5 test questions against the 57-actor operational state:
   - "What if BRICS abandons the dollar?"
   - "How would a NATO-Russia confrontation over the Baltics unfold?"
   - "What happens if there is a drought crisis in Sub-Saharan Africa?"
   - "What if North Korea tests a nuclear device?"
   - "How would OPEC production cuts affect global stability?"
2. Record the resolved actor list and unresolved list for each.
3. Count how many of the key implied actors were correctly resolved.

**Pass criterion.** At least 3 of the 5 questions should resolve the majority of their geopolitically correct actor set (≥ 60% of the actors a domain expert would choose). The fallback should never be triggered silently for questions that clearly name geopolitical entities.

**Recommendation if fail.** Expand the alias table to cover all major geopolitical groupings (EU → top EU members, NATO → US + key European allies, BRICS → Brazil/Russia/India/China/South Africa, OPEC → Saudi Arabia + key producers). Add a confidence flag to the scenario output that distinguishes "actors explicitly mentioned" from "actors inferred by fallback." Consider a lightweight LLM call for actor inference when string matching fails.

---

### Q10. Game-Theory Equilibrium Search — Hedge Algorithm Convergence Guarantees

**What to inspect.** In `equilibrium_runner.py`, the equilibrium search uses a multiplicative-weights (Hedge) algorithm with `eta=0.1`, `exploration_eps=0.1`, and `max_episodes=50`. Convergence is checked by `history.has_converged(threshold=0.02)` on external regret. The correlated equilibrium is solved by LP in `correlated_eq.py`.

**Why it matters.** (a) **50 episodes may be insufficient.** The theoretical regret bound for multiplicative weights is `O(√(T ln N))`. With T=50 and N≈10 actions, the average regret bound is `√(50 × ln(10))/50 ≈ 0.21`, which is 10× the convergence threshold of 0.02. The algorithm may declare non-convergence even when it would converge with more episodes. (b) **Exploration-exploitation conflict.** With `exploration_eps=0.1`, 10% of episodes are random. In a 50-episode run, ~5 episodes are pure noise, which inflates the empirical regret and can prevent convergence detection. (c) **Stage-game reuse.** The comment says "the stage game is static for a fixed scenario, so reuse the same payoff matrix across episodes." This is correct for a normal-form game, but the game runner's scoring depends on the `world` state which is not being stepped during the equilibrium search. If the intention is to find an equilibrium of the static game, this is fine. But if the intention is to model a sequential game (3-year horizon), the static approach may miss subgame dynamics.

**How to carry out.**

1. Take the bundled `maritime_pressure_game.json` test case.
2. Run `run_equilibrium_search()` with `max_episodes=50` and record whether it converges.
3. Re-run with `max_episodes=200` and compare the recommended profile and regret values.
4. Re-run with `exploration_eps=0.0` (pure exploitation) at 50 episodes and check if convergence improves.
5. Report the gap between the LP correlated equilibrium and the Hedge empirical CCE.

**Pass criterion.** The recommended profile should be stable across the 50-episode and 200-episode runs (same dominant action for each player). The mean external regret at 200 episodes should be < 0.02.

**Recommendation if fail.** Increase default `max_episodes` to at least 200 (computational cost is minimal since it reuses the cached payoff matrix). Reduce `exploration_eps` to 0.03 after the first 20 episodes (decaying exploration). Add an explicit convergence diagnostic to the output artifact so users can see whether the equilibrium is well-identified.

---

## Part B — Quantitative Tests

Each quantitative test specifies exact inputs, computations, metrics, and acceptance thresholds. All tests should be runnable with `POLICY_MODE=simple`, `DISABLE_EXTREME_EVENTS=1` (unless noted), and `SIM_SEED=42` for reproducibility. No LLM API calls are required.

---

### T1. GDP Trajectory Stability Under Initial-Condition Perturbation (Lyapunov-Type Test)

**Objective.** Measure the sensitivity of the model's 5-year GDP trajectory to small perturbations of the initial state, distinguishing between legitimate amplification (e.g., crisis thresholds) and numerical instability.

**Setup.**

- Load the 57-actor operational state (`agent_states_operational.csv`).
- Select 10 representative agents: the top-5 by GDP and the bottom-5 by GDP (among the 57).
- Define the baseline run: `SIM_YEARS=5, POLICY_MODE=simple, DISABLE_EXTREME_EVENTS=1, SIM_SEED=42`.

**Perturbation design.** For each of the 10 agents, independently perturb each of the following 6 initial-state variables by ±1%: `gdp, capital, population, trust_gov, social_tension, public_debt`. This gives 10 agents × 6 variables × 2 directions = 120 perturbed runs (each run perturbs exactly one variable for one agent; all other agents keep their baseline values).

**Metric computation.** For each perturbed run `p` and each agent `a`, compute:

```
GDP_divergence(a, p, t) = |GDP_perturbed(a,t) − GDP_baseline(a,t)| / GDP_baseline(a,t)
```

Then compute the maximum divergence at t=5 across all perturbations for each agent:

```
MaxDiv(a) = max over p of GDP_divergence(a, p, t=5)
```

And the perturbation amplification ratio:

```
AmpRatio(a, p) = GDP_divergence(a, p, t=5) / perturbation_magnitude(0.01)
```

**Reporting.** Produce a table: `agent_name | perturbed_variable | direction | GDP_divergence_t5 | AmpRatio`. Flag any row where `AmpRatio > 10` (i.e., a 1% input perturbation causes >10% output divergence in 5 years).

**Acceptance thresholds.**

- For agents not near a crisis threshold: `AmpRatio < 5` for all 6 variables.
- For agents near a crisis threshold (debt/GDP > 1.0 or trust < 0.25): `AmpRatio < 15` is acceptable, but must be documented as threshold-driven amplification.
- Global: No agent should have `AmpRatio > 20` for any perturbation.

**Computational cost.** 120 perturbed runs × 5 years × 57 agents = ~34,200 agent-steps. At ~1ms per step, this completes in under 1 minute with no LLM calls.

---

### T2. Parameter Sensitivity Analysis: Prior-Heavy Calibration Parameters

**Objective.** Quantify the marginal effect of the 8 most uncertain calibration parameters on the model's key output metrics, to identify which priors most urgently need empirical calibration.

**Setup.**

- Load the 20-actor compact state (`agent_states.csv`).
- Baseline run: `SIM_YEARS=5, POLICY_MODE=simple, DISABLE_EXTREME_EVENTS=1, SIM_SEED=42`.

**Parameter selection.** The 8 parameters, chosen because they are tagged `[PRIOR]` and appear in calibration-sensitive feedback loops:

| # | Parameter | Baseline | Perturbation range | Module |
|---|-----------|----------|-------------------|--------|
| 1 | `GINI_FISCAL_SENS` | −60.0 | [−90.0, −30.0] | social |
| 2 | `CRISK_TEMP_SENSITIVITY` | 0.45 | [0.20, 0.70] | climate |
| 3 | `STRUCTURAL_TRANSITION_POLICY_SENS` | 0.50 | [0.25, 0.75] | climate |
| 4 | `DAMAGE_QUAD_COEFF` | 0.006 | [0.003, 0.012] | climate |
| 5 | `TRUST_TENSION_SENS` | −0.08 | [−0.15, −0.04] | social |
| 6 | `DEBT_SPREAD_QUADRATIC` | 0.10 | [0.05, 0.20] | economy |
| 7 | `EVENT_MAX_EXTRA_PROB` | 0.07 | [0.03, 0.14] | climate |
| 8 | `SAVINGS_STABILITY_SENS` | 0.60 | [0.30, 0.90] | economy |

**Sweep design.** For each parameter, evaluate at 5 equally-spaced points within the perturbation range (including endpoints and baseline). Total runs: 8 parameters × 5 values = 40 runs.

**Metric computation.** For each run, compute:

- `M1`: Global GDP at t=5 (sum over all agents)
- `M2`: Global average trust at t=5
- `M3`: Global average social_tension at t=5
- `M4`: Global CO2 concentration at t=5
- `M5`: Number of agents in credit_zone ∈ {distressed, default} at t=5

For each parameter `θ_i`, compute the normalized sensitivity index:

```
S(θ_i, M_j) = [M_j(θ_max) − M_j(θ_min)] / M_j(baseline)
```

And the elasticity at baseline:

```
E(θ_i, M_j) = (ΔM_j / M_j_baseline) / (Δθ_i / θ_i_baseline)
```

**Reporting.** Produce a 8×5 sensitivity matrix (parameters × metrics). Rank parameters by the maximum absolute sensitivity index across all metrics. Highlight any parameter where `|S| > 0.15` (i.e., the parameter range produces >15% variation in any output metric).

**Acceptance thresholds.**

- No parameter should have `|S| > 0.30` for GDP or CO2 (model should not be fragile to ±50% variations of a single prior).
- At least 3 of the 8 parameters should have `|S| > 0.05` for at least one metric (confirming that the model is responsive to its priors, not degenerate).
- The top-ranked parameter should be flagged as the highest-priority target for the next empirical calibration round.

**Computational cost.** 40 runs × 5 years × 20 agents = 4,000 agent-steps. Under 10 seconds.

---

### T3. Crisis-Pathway Replication Test: Debt Crisis Duration and GDP Impact

**Objective.** Verify that the model can reproduce the stylized trajectory of a canonical sovereign debt crisis (Argentina 2001–2006: ~5-year crisis duration, ~20% cumulative GDP loss, ~30 Gini-point increase in inequality) and a canonical regime crisis (Turkey 2018: FX crisis, ~15% GDP per-capita loss, resolution within 2 years).

**Setup.**

- Use the operational_v2 calibration cases: `argentina_default_2001.yaml` and `turkey_fx_crisis_2018.yaml`.
- Run through the `calibration.py:run_operational_calibration()` harness with `n_runs=3` (stochastic averaging).

**Metric computation for Argentina.**

- `D1`: Number of consecutive years with `debt_crisis_active_years > 0` (target: ≥ 4).
- `D2`: Cumulative GDP loss from peak to trough, as % of peak GDP (target: 15–25%).
- `D3`: Peak Gini increase from baseline (target: 5–15 Gini points).
- `D4`: Years to GDP recovery (return to pre-crisis level) (target: 4–7 years).

**Metric computation for Turkey.**

- `D5`: Duration of the acute FX/debt stress episode (target: 1–3 years).
- `D6`: Maximum single-year GDP per-capita loss (target: 10–20%).
- `D7`: Trust floor during crisis (target: > 0.15 — should not trigger regime collapse).

**Reporting.** For each metric, report: model value (mean ± std across 3 runs), target range, and pass/fail.

**Acceptance thresholds.**

- Argentina: At least 3 of 4 metrics within target range.
- Turkey: At least 2 of 3 metrics within target range.
- If the debt crisis clears in ≤ 2 years for Argentina (as suggested by the audit), this test should explicitly flag the Q5 onset-haircut paradox as confirmed.

**Computational cost.** 2 cases × 3 runs × ~8 years × ~10 agents = ~480 agent-steps. Under 5 seconds.

---

### T4. Bilateral Relation Asymmetry and Irreversibility Audit

**Objective.** Quantify whether the bilateral relation dynamics exhibit pathological asymmetries (e.g., a → b trust ≠ b → a trust without justification) and whether trade intensity suffers from the irreversible decay identified in Q7.

**Setup.**

- Load the 20-actor compact state.
- Run `SIM_YEARS=10, POLICY_MODE=simple, DISABLE_EXTREME_EVENTS=1, SIM_SEED=42`.
- Record the full relation matrix at t=0, t=5, and t=10.

**Metric computation.**

**Asymmetry metric.** For each directed pair (a, b) at each snapshot, compute:

```
trust_asym(a,b,t) = |trust(a→b,t) − trust(b→a,t)|
conflict_asym(a,b,t) = |conflict(a→b,t) − conflict(b→a,t)|
trade_asym(a,b,t) = |trade_intensity(a→b,t) − trade_intensity(b→a,t)|
```

Report the distribution (mean, max, 90th percentile) of each asymmetry metric at t=0, t=5, t=10.

**Irreversibility metric.** Compute:

```
trade_decay(a,b) = trade_intensity(a→b, t=10) / trade_intensity(a→b, t=0)
```

Report the distribution of `trade_decay` across all pairs. Flag the fraction of pairs where `trade_decay < 0.5` (trade intensity halved in 10 years).

**Mean-reversion check.** For trust and conflict, compute:

```
trust_drift(a,b) = trust(a→b, t=10) − trust(a→b, t=0)
conflict_drift(a,b) = conflict(a→b, t=10) − conflict(a→b, t=0)
```

Report the mean drift. Trust should drift toward 0.6 and conflict toward 0.1 (the hardcoded baselines in `update_relations_endogenous()`).

**Acceptance thresholds.**

- Trust asymmetry at t=10: mean < 0.10, max < 0.25.
- Trade intensity: median `trade_decay` > 0.70 (no more than 30% decay over 10 years in a peaceful baseline).
- Fraction of pairs with `trade_decay < 0.5`: < 10%.
- Mean trust drift should be directed toward 0.6 (positive if initial trust < 0.6, negative if > 0.6).

**Computational cost.** 1 run × 10 years × 20 agents = 200 agent-steps. Under 2 seconds. Relation matrix has 20×19 = 380 directed pairs.

---

### T5. Outcome-Probability Calibration: Discrimination and Calibration Score Under Historical Near-Miss Cases

**Objective.** Verify that the scenario-evaluation layer produces outcome distributions that (a) correctly discriminate between crisis and stable scenarios, and (b) assign calibrated probability masses that pass a simplified Brier-type reliability check against the operational_v1 and operational_v2 case suites.

**Setup.**

- Load the 57-actor operational state.
- Process all 11 `operational_v1` cases and all 5 `operational_v2` cases through the `GameRunner.evaluate_scenario()` pipeline.

**Metric computation.**

**Discrimination metric (M1).** For each case, record the dominant outcome label (highest-probability outcome). Compare against the expected label from the case definition (stored in the YAML/JSON files). Compute:

```
accuracy = (number of cases where dominant label matches expected) / (total cases)
```

**Criticality separation (M2).** Compute the mean `criticality_score` separately for the 7 crisis cases and the 4 stable controls in operational_v1:

```
gap = mean_criticality(crisis_cases) − mean_criticality(stable_controls)
```

This should be strictly positive and > 0.15.

**Probability mass allocation (M3).** For each crisis case, compute the total probability mass on all non-status_quo outcomes. For each stable control, compute the same. Report:

```
crisis_avg_non_sq_mass = mean(non_sq_mass for crisis cases)
stable_avg_non_sq_mass = mean(non_sq_mass for stable cases)
```

**Calibration consistency (M4).** For the 5 `operational_v2` near-miss cases (Brazil, South Korea, Turkey, Argentina, France), verify that the dominant label matches the expected near-miss interpretation: Brazil/South Korea → negotiated_deescalation, Turkey/Argentina → internal_destabilization, France → status_quo.

**Sensitivity stability (M5).** For each case, perturb the top-3 most sensitive outcome weights (identified in the existing `geo_sensitivity_operational_v1.json` report) by ±10% and check whether the dominant label flips. Count flips:

```
flip_rate = (number of label flips) / (total perturbations)
```

**Reporting.** Produce a table: `case_id | expected_label | predicted_label | match | criticality_score | non_sq_mass | calibration_score`.

**Acceptance thresholds.**

- `accuracy` (M1) ≥ 0.80 across all 16 cases.
- `gap` (M2) > 0.15.
- `crisis_avg_non_sq_mass` (M3) > 0.40; `stable_avg_non_sq_mass` < 0.30.
- M4: At least 4 of 5 near-miss cases match expected label.
- `flip_rate` (M5) < 0.15 (no more than 15% of ±10% perturbations should flip the dominant label).

**Computational cost.** 16 base evaluations + ~96 perturbation evaluations = ~112 scenario evaluations. Each evaluation is a matrix computation (no simulation steps), completing in under 5 seconds total.

---

## Appendix A — Implementation Notes for the Coder

**Environment requirements.** Python 3.10+, the GIM-14 `gim` package installed in editable mode (`pip install -e .`). No external API keys needed (set `NO_LLM=1`). Optional: `scipy` for the correlated-equilibrium LP in T5.

**Shared test harness.** All quantitative tests should use a shared fixture loader:

```python
from gim.core.world_factory import make_world_from_csv
from gim.core.simulation import step_world
from gim.core.policy import make_policy_map

def load_world(state_csv="agent_states_operational.csv"):
    return make_world_from_csv(state_csv)

def run_baseline(world, years=5, seed=42):
    import os, copy
    os.environ["SIM_SEED"] = str(seed)
    os.environ["DISABLE_EXTREME_EVENTS"] = "1"
    policies = make_policy_map(world.agents.keys(), mode="simple")
    memory = {}
    w = copy.deepcopy(world)
    trajectory = [copy.deepcopy(w)]
    for _ in range(years):
        w = step_world(w, policies, enable_extreme_events=False)
        trajectory.append(copy.deepcopy(w))
    return trajectory
```

**Parameter perturbation helper.** For T2, monkey-patch `calibration_params` at runtime:

```python
import gim.core.calibration_params as cal

def with_param(name, value):
    original = getattr(cal, name)
    setattr(cal, name, value)
    return original  # caller restores after run
```

**Output format.** Each test should produce a JSON artifact with the structure:

```json
{
  "test_id": "T1",
  "timestamp": "2026-03-16T...",
  "status": "PASS" | "FAIL" | "WARN",
  "metrics": { ... },
  "flagged_items": [ ... ],
  "notes": "..."
}
```

---

## Appendix B — Test Priority and Execution Order

| Priority | Test | Estimated runtime | Dependencies |
|----------|------|-------------------|-------------|
| 1 (critical) | T3 | 5 sec | operational_v2 YAML cases |
| 2 (critical) | Q5 | manual review + 30 sec code | social.py |
| 3 (high) | T1 | 60 sec | operational CSV |
| 4 (high) | T2 | 10 sec | compact CSV |
| 5 (high) | Q2 | manual review + 5 sec code | climate.py |
| 6 (high) | T5 | 5 sec | calibration cases |
| 7 (medium) | Q1 | manual review + 30 sec code | economy.py |
| 8 (medium) | T4 | 2 sec | compact CSV |
| 9 (medium) | Q4 | 30 sec code | simulation.py |
| 10 (medium) | Q6 | manual review | game_runner.py |
| 11 (medium) | Q7 | 5 sec code | political_dynamics.py |
| 12 (medium) | Q8 | 10 sec code | economy.py, credit_rating.py |
| 13 (lower) | Q3 | manual review (or 1 LLM call) | policy.py |
| 14 (lower) | Q9 | 5 sec code | scenario_compiler.py |
| 15 (lower) | Q10 | 30 sec code | equilibrium_runner.py |

Total estimated computational time for all non-LLM tests: < 3 minutes on a single core, no GPU, no API calls.

---

## Part C — LLM-Policy Validation Tests

The tests in Parts A and B use `POLICY_MODE=simple`, which exercises only the deterministic heuristic policy. GIM-14's core use case is LLM-driven agent decision-making, where DeepSeek receives observation JSON and returns full policy actions for each of the 57 agents each year. The following 6 tests validate the LLM integration path specifically. They require a valid `DEEPSEEK_API_KEY` and will consume approximately 600–1200 API calls total (~$2–5).

All LLM tests use a 10-agent world (the compact `agent_states.csv`) to keep costs proportional. A 3-year run with 10 agents costs 30 LLM calls per run.

---

### Q11. LLM Behavioral Heterogeneity Under Full Simulation Trajectory

**What to inspect.** Q3 analyzed the LLM prompt in isolation. Q11 runs a full 3-year LLM-driven simulation and then measures whether structurally different countries (wealthy democracies vs. fragile autocracies vs. resource exporters) actually pursue distinguishable policy trajectories over time.

**Why it matters.** Even if the LLM produces varied one-shot responses (Q3), the simulation's political-constraint filter (`apply_political_constraints`) and endogenous dynamics may homogenize outcomes over multiple steps. The real question is whether archetype differences survive the full feedback loop.

**How to carry out.**

1. Load the 10-agent compact world. Classify agents into 3 archetypes by GDP per capita and regime_type: (A) high-income democracies, (B) middle-income / hybrid, (C) low-income / autocratic.
2. Run a 3-year simulation with `POLICY_MODE=llm`, recording the action log.
3. For each archetype group, compute the group-average over all steps of: `climate_policy` distribution (none/weak/moderate/strong counts), `military_spending_change`, `social_spending_change`, number of sanctions issued, number of trade deals proposed.
4. Compute pairwise inter-archetype distance as L1 on the normalized action profile vectors. Compute intra-archetype distance as the mean L1 between individual agents and their group centroid.

**Metrics.**
- `inter_archetype_distance`: mean L1 between archetype centroids.
- `intra_archetype_distance`: mean L1 within archetypes.
- `discrimination_ratio`: inter / intra (should be > 1.5).

**Acceptance threshold.** `discrimination_ratio > 1.5`. At least 2 of the 3 archetypes should have a distinct modal `climate_policy` level.

---

### Q12. LLM Fallback Rate and Silent Degradation

**What to inspect.** In `simulation.py`, when an LLM policy call fails (timeout, JSON parse error, schema violation), the model silently falls back to `simple_rule_based_policy`. The fallback is logged at WARNING level but does not appear in the action log or the output artifacts.

**Why it matters.** If the fallback rate is high (>10%), the simulation is effectively running in hybrid mode where some agents use sophisticated LLM reasoning and others use the minimal heuristic. This creates an asymmetric intelligence landscape: LLM agents will sanction, trade, and escalate, while fallback agents sit passively. The user has no way to detect this from the output.

**How to carry out.**

1. Load the 10-agent compact world.
2. Wrap the `llm_policy` function with a counting decorator that records: total calls, successful JSON parses, fallback triggers, and the exception type for each failure.
3. Run a 3-year simulation with `POLICY_MODE=llm`.
4. Compute `fallback_rate = fallback_count / total_calls`.
5. For agents that experienced fallback, compare their GDP trajectory to agents that never fell back.

**Metrics.**
- `total_llm_calls`: expected ≈ 30 (10 agents × 3 years).
- `fallback_count`: number of calls that fell back to simple policy.
- `fallback_rate`: ratio.
- `fallback_agents`: list of agent IDs that experienced at least one fallback.
- `gdp_gap_fallback_vs_llm`: mean GDP growth difference between fallback and non-fallback agents.

**Acceptance threshold.** `fallback_rate < 0.10`. No single agent should fall back more than once in a 3-year run.

---

### T6. LLM Sanity Bounds: Plausible Trajectories Under LLM Policies

**Objective.** Verify that LLM-driven policies do not push the model into implausible states: GDP should not collapse without a crisis trigger, trust should not go to 0 for stable countries, no agent should accumulate debt/GDP > 3.0 without a debt crisis active.

**Setup.**
- Load the 10-agent compact world.
- Run `SIM_YEARS=3, POLICY_MODE=llm, SIM_SEED=42`.
- Record full trajectories for all agents.

**Metric computation.** For each agent at each step, check:

- `B1`: GDP change > −20% per year (no single-year collapse without war/crisis).
- `B2`: `trust_gov` stays in [0.05, 1.0] (should not hit machine-zero).
- `B3`: `social_tension` stays in [0.0, 0.95] (should not saturate at 1.0 for stable agents).
- `B4`: `debt/GDP` < 3.0 unless `debt_crisis_active_years > 0`.
- `B5`: Global temperature change < 0.5°C over 3 years (no runaway from LLM policy).
- `B6`: No agent population drops > 5% in a single year (without war).

Count the total bound violations across all agents and steps.

**Acceptance threshold.** Total violations < 3 across all agents and steps (allowing for occasional threshold-crossing from compound stress). Zero `B1` violations for agents starting in stable conditions (trust > 0.5, tension < 0.3).

**Computational cost.** 1 run × 3 years × 10 agents = 30 LLM calls.

---

### T7. LLM Reproducibility: Trajectory Stability Across Identical Runs

**Objective.** Measure whether two runs with identical initial state and seed produce similar macro trajectories despite the inherent stochasticity of LLM outputs at temperature=0.2.

**Setup.**
- Load the 10-agent compact world.
- Run A: `SIM_YEARS=3, POLICY_MODE=llm, SIM_SEED=42`.
- Run B: `SIM_YEARS=3, POLICY_MODE=llm, SIM_SEED=42` (identical).
- Both runs use the same world state, same seed, same everything.

**Metric computation.** For each agent at t=3, compute:

```
gdp_divergence(a) = |GDP_A(a,3) − GDP_B(a,3)| / mean(GDP_A(a,3), GDP_B(a,3))
trust_divergence(a) = |trust_A(a,3) − trust_B(a,3)|
tension_divergence(a) = |tension_A(a,3) − tension_B(a,3)|
```

Also compare action profiles: for each agent at each step, check whether the dominant action label (from `_policy_record_label`) matches between runs.

```
action_match_rate = (matched action labels) / (total agent-steps)
```

**Acceptance thresholds.**
- Mean `gdp_divergence` < 0.08 (8% — LLM nondeterminism should not cause >8% GDP divergence in 3 years).
- Mean `trust_divergence` < 0.10.
- `action_match_rate` > 0.60 (at least 60% of agent-steps choose the same dominant action category).

**Computational cost.** 2 runs × 30 LLM calls = 60 LLM calls.

---

### T8. LLM Policy Coherence: Context-Appropriate Actions

**Objective.** Verify that LLM-generated policies are contextually coherent — that agents in crisis adopt crisis-appropriate policies and stable agents don't randomly escalate.

**Setup.**
- Load the 10-agent compact world.
- Run `SIM_YEARS=3, POLICY_MODE=llm, SIM_SEED=42`, recording the action log.

**Coherence rules to check.** For each agent at each step, evaluate:

- `C1 — Crisis response`: If an agent has `debt_stress > 0.5` or `tension > 0.6`, it should NOT increase `military_spending` above +0.005 (should prioritize social spending or austerity). Violation if it does.
- `C2 — Security awareness`: If an agent has `security_margin < 1.0` AND a neighbor with `conflict_level > 0.5`, it should have `military_spending_change ≥ 0` or a security action other than "none". Violation if military spending decreases while under threat.
- `C3 — Climate-income alignment`: If `gdp_per_capita < 5000`, the agent should not choose `climate_policy = "strong"`. If `gdp_per_capita > 40000` and `climate_risk > 0.3`, the agent should not choose `climate_policy = "none"` for all 3 years.
- `C4 — No self-sanction`: No agent should sanction itself or a close ally (same `alliance_block`) without `conflict_level > 0.5` with that ally.
- `C5 — Explanation non-empty`: Every LLM action should have a non-empty `explanation` field (confirms the LLM actually generated a rationale, not just JSON).

**Metric computation.** Count violations per rule. Compute `coherence_score = 1 − (total_violations / total_agent_steps)`.

**Acceptance threshold.** `coherence_score > 0.85`. No single rule should have violations > 30% of applicable agent-steps.

**Computational cost.** 0 additional LLM calls (reuses T6 or runs its own single 30-call run).

---

### T9. LLM vs Simple Policy Divergence: Does the LLM Add Value?

**Objective.** Compare 3-year trajectories under LLM policies versus simple policies to confirm that LLM decision-making materially changes outcomes. If the two modes produce nearly identical trajectories, the LLM is not contributing meaningful strategic reasoning and its cost is unjustified.

**Setup.**
- Load the 10-agent compact world.
- Run A: `SIM_YEARS=3, POLICY_MODE=simple, SIM_SEED=42`.
- Run B: `SIM_YEARS=3, POLICY_MODE=llm, SIM_SEED=42`.

**Metric computation.**

For each agent at t=3:
```
gdp_delta(a) = GDP_llm(a,3) − GDP_simple(a,3)
trust_delta(a) = trust_llm(a,3) − trust_simple(a,3)
tension_delta(a) = tension_llm(a,3) − tension_simple(a,3)
```

Aggregate:
- `mean_abs_gdp_delta`: mean |gdp_delta| across agents, normalized by baseline GDP.
- `mean_abs_trust_delta`: mean |trust_delta| across agents.
- `action_diversity_ratio`: count of unique (agent, action_label) pairs in LLM run vs simple run.
- `sanctions_issued_llm`: total sanctions in LLM run.
- `sanctions_issued_simple`: total sanctions in simple run (expected: 0).
- `trade_deals_llm`: total trade deals in LLM run.
- `trade_deals_simple`: total trade deals in simple run (expected: 0).

**Acceptance thresholds.**
- `mean_abs_gdp_delta > 0.02` (LLM should cause at least 2% GDP divergence from simple over 3 years — if not, LLM is essentially a no-op).
- `action_diversity_ratio > 2.0` (LLM should produce at least 2× more unique action types).
- `sanctions_issued_llm > 0` (LLM should produce at least some foreign-policy activity).

**Computational cost.** 1 LLM run (30 calls) + 1 simple run (0 calls) = 30 LLM calls.

---

## Appendix C — LLM Test Priority and Cost Estimate

| Priority | Test | Estimated LLM calls | Estimated time | Key dependency |
|----------|------|---------------------|---------------|----------------|
| 1 (critical) | T6 | 30 | 2–3 min | DEEPSEEK_API_KEY |
| 2 (critical) | T7 | 60 | 4–5 min | DEEPSEEK_API_KEY |
| 3 (high) | T9 | 30 | 2–3 min | DEEPSEEK_API_KEY |
| 4 (high) | T8 | 0 (reuses T6) | <10 sec | action log from T6 |
| 5 (medium) | Q11 | 30 | 2–3 min | DEEPSEEK_API_KEY |
| 6 (medium) | Q12 | 30 | 2–3 min | DEEPSEEK_API_KEY |

Total estimated LLM calls: ~180–210. At DeepSeek pricing (~$0.002/call), total cost: ~$0.40–0.50.
Total estimated wall time for LLM tests: 10–15 minutes (dominated by API latency).
