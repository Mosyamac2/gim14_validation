"""
T4 — Bilateral Relation Asymmetry and Irreversibility Audit

Measure trust/conflict asymmetry, trade-intensity decay, and
mean-reversion behaviour over a 10-year baseline run.
"""
from __future__ import annotations
import statistics
from ..helpers import (
    gim, load_compact_world, run_steps,
    make_result, determine_status, Timer,
)


def _relation_snapshot(world) -> dict:
    """Extract full relation matrix as a dict of (from,to) -> metrics."""
    snap = {}
    for from_id, rels in world.relations.items():
        for to_id, rel in rels.items():
            snap[(from_id, to_id)] = {
                "trust": rel.trust,
                "conflict": rel.conflict_level,
                "trade_intensity": rel.trade_intensity,
                "trade_barrier": rel.trade_barrier,
            }
    return snap


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        world = load_compact_world()
        traj = run_steps(world, years=10, seed=42)

        snap_t0 = _relation_snapshot(traj[0])
        snap_t5 = _relation_snapshot(traj[5])
        snap_t10 = _relation_snapshot(traj[10])

        pairs = list(snap_t0.keys())
        n_pairs = len(pairs)

        # ------------------------------------------------------------------
        # Asymmetry metrics at each snapshot
        # ------------------------------------------------------------------
        def compute_asymmetry(snap, field):
            vals = []
            seen = set()
            for (a, b) in pairs:
                if (b, a) in snap and (b, a) not in seen:
                    vals.append(abs(snap[(a, b)][field] - snap[(b, a)][field]))
                    seen.add((a, b))
            return vals

        asym_results = {}
        for label, snap in [("t0", snap_t0), ("t5", snap_t5), ("t10", snap_t10)]:
            row = {}
            for field in ["trust", "conflict", "trade_intensity"]:
                vals = compute_asymmetry(snap, field)
                if vals:
                    row[f"{field}_mean"] = round(statistics.mean(vals), 4)
                    row[f"{field}_max"] = round(max(vals), 4)
                    row[f"{field}_p90"] = round(sorted(vals)[int(0.9 * len(vals))], 4)
                else:
                    row[f"{field}_mean"] = 0.0
                    row[f"{field}_max"] = 0.0
                    row[f"{field}_p90"] = 0.0
            asym_results[label] = row

        # ------------------------------------------------------------------
        # Trade intensity decay (irreversibility metric)
        # ------------------------------------------------------------------
        trade_decays = []
        for pair in pairs:
            ti_0 = snap_t0[pair]["trade_intensity"]
            ti_10 = snap_t10[pair]["trade_intensity"]
            if ti_0 > 1e-6:
                trade_decays.append(ti_10 / ti_0)
            else:
                trade_decays.append(1.0)

        median_decay = sorted(trade_decays)[len(trade_decays) // 2]
        frac_below_05 = sum(1 for d in trade_decays if d < 0.5) / max(len(trade_decays), 1)

        # ------------------------------------------------------------------
        # Mean-reversion check for trust and conflict
        # ------------------------------------------------------------------
        trust_drifts = []
        conflict_drifts = []
        for pair in pairs:
            trust_drifts.append(snap_t10[pair]["trust"] - snap_t0[pair]["trust"])
            conflict_drifts.append(snap_t10[pair]["conflict"] - snap_t0[pair]["conflict"])

        mean_trust_drift = statistics.mean(trust_drifts) if trust_drifts else 0
        mean_conflict_drift = statistics.mean(conflict_drifts) if conflict_drifts else 0

        # Trust baseline = 0.6 (initialized), so most start at 0.6 and drift is expected ~0
        # Conflict baseline = 0.1 (initialized at 0.1), mean-reversion keeps it stable

        # ------------------------------------------------------------------
        # Assess
        # ------------------------------------------------------------------
        trust_asym_ok = asym_results["t10"]["trust_mean"] < 0.10
        trust_asym_max_ok = asym_results["t10"]["trust_max"] < 0.25
        trade_median_ok = median_decay > 0.70
        trade_extreme_ok = frac_below_05 < 0.10

        conditions = {
            "trust_asym_mean_lt_010": trust_asym_ok,
            "trust_asym_max_lt_025": trust_asym_max_ok,
            "trade_median_decay_gt_070": trade_median_ok,
            "trade_extreme_decay_lt_10pct": trade_extreme_ok,
        }

    return make_result(
        test_id="T4",
        status=determine_status(conditions),
        metrics={
            "n_pairs": n_pairs,
            "trust_asym_t10_mean": asym_results["t10"]["trust_mean"],
            "trust_asym_t10_max": asym_results["t10"]["trust_max"],
            "trade_median_decay_ratio": round(median_decay, 4),
            "trade_frac_below_05": round(frac_below_05, 3),
            "mean_trust_drift_10y": round(mean_trust_drift, 5),
            "mean_conflict_drift_10y": round(mean_conflict_drift, 5),
        },
        flagged_items=[
            f"Trade median decay = {median_decay:.3f} (threshold: >0.70)"
            if not trade_median_ok else "",
            f"Trust asymmetry max = {asym_results['t10']['trust_max']:.3f}"
            if not trust_asym_max_ok else "",
        ],
        notes=f"Completed in {t.elapsed:.1f}s. {n_pairs} directed pairs.",
        details={
            "asymmetry_by_snapshot": asym_results,
            "trade_decay_distribution": {
                "min": round(min(trade_decays), 4),
                "p10": round(sorted(trade_decays)[int(0.1*len(trade_decays))], 4),
                "median": round(median_decay, 4),
                "p90": round(sorted(trade_decays)[int(0.9*len(trade_decays))], 4),
                "max": round(max(trade_decays), 4),
            },
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
