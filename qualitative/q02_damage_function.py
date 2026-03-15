"""
Q2 — Climate Damage Function Shape at Model-Relevant Temperatures

Evaluate the damage multiplier curve at model-relevant temperatures
and compare against literature references.
"""
from __future__ import annotations
from ..helpers import gim, load_compact_world, run_steps, make_result, determine_status, Timer


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        g = gim()
        from gim.core.climate import climate_damage_multiplier, effective_damage_multiplier
        from gim.core.core import TGLOBAL_2023_C
        cal = g["cal"]

        # ------------------------------------------------------------------
        # Step 1: Evaluate damage curve at key temperatures
        # ------------------------------------------------------------------
        anomalies = [1.0, 1.29, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        curve = []
        for T in anomalies:
            mult = climate_damage_multiplier(T)
            gdp_loss_pct = (1.0 - mult) * 100
            curve.append({
                "temperature_anomaly": T,
                "damage_multiplier": round(mult, 6),
                "gdp_loss_pct": round(gdp_loss_pct, 3),
            })

        # Find peak benefit and zero-crossing
        fine_temps = [TGLOBAL_2023_C + i * 0.05 for i in range(-10, 60)]
        peak_benefit_temp = max(fine_temps, key=lambda T: climate_damage_multiplier(T))
        peak_benefit_val = climate_damage_multiplier(peak_benefit_temp)

        zero_crossing = None
        for i in range(len(fine_temps) - 1):
            m1 = climate_damage_multiplier(fine_temps[i])
            m2 = climate_damage_multiplier(fine_temps[i + 1])
            if m1 >= 1.0 and m2 < 1.0:
                zero_crossing = round(fine_temps[i], 2)
                break

        # ------------------------------------------------------------------
        # Step 2: Check damage at +3°C against literature (DICE ~2-5%)
        # ------------------------------------------------------------------
        damage_at_3c = 1.0 - climate_damage_multiplier(3.0)
        damage_at_3c_pct = damage_at_3c * 100

        # ------------------------------------------------------------------
        # Step 3: Run with and without benefit gaussian
        # ------------------------------------------------------------------
        world = load_compact_world()

        traj_baseline = run_steps(world, years=10, seed=42, enable_extreme_events=False)
        baseline_gdp_t10 = sum(a.economy.gdp for a in traj_baseline[10].agents.values())

        from ..helpers import patch_param
        with patch_param("DAMAGE_BENEFIT_MAX", 0.0):
            traj_nobenefit = run_steps(world, years=10, seed=42, enable_extreme_events=False)
        nobenefit_gdp_t10 = sum(a.economy.gdp for a in traj_nobenefit[10].agents.values())

        gdp_diff_pct = (baseline_gdp_t10 - nobenefit_gdp_t10) / max(nobenefit_gdp_t10, 1e-12) * 100

        # ------------------------------------------------------------------
        # Assess
        # ------------------------------------------------------------------
        damage_3c_ok = damage_at_3c_pct >= 2.0  # Should be at least 2% at +3C
        no_warming_preference = gdp_diff_pct < 5.0  # Benefit shouldn't dominate

        conditions = {
            "damage_at_3C_ge_2pct": damage_3c_ok,
            "no_warming_preference": no_warming_preference,
        }

    return make_result(
        test_id="Q2",
        status=determine_status(conditions),
        metrics={
            "damage_at_3C_pct": round(damage_at_3c_pct, 3),
            "peak_benefit_temperature": round(peak_benefit_temp, 3),
            "peak_benefit_multiplier": round(peak_benefit_val, 6),
            "zero_crossing_temperature": zero_crossing,
            "gdp_diff_benefit_vs_no_pct": round(gdp_diff_pct, 3),
            "DAMAGE_QUAD_COEFF": cal.DAMAGE_QUAD_COEFF,
            "DAMAGE_BENEFIT_MAX": cal.DAMAGE_BENEFIT_MAX,
            "DAMAGE_BENEFIT_PEAK": cal.DAMAGE_BENEFIT_PEAK,
        },
        flagged_items=[
            f"damage_at_3C = {damage_at_3c_pct:.2f}% (literature expects 2-5%)"
            if not damage_3c_ok else "",
        ],
        notes=f"Completed in {t.elapsed:.1f}s",
        details={
            "damage_curve": curve,
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
