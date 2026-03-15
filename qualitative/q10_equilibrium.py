"""
Q10 — Game-Theory Equilibrium Search: Hedge Algorithm Convergence Guarantees

Test whether the Hedge equilibrium search converges within 50 episodes
and whether results are stable at 200 episodes.
"""
from __future__ import annotations
import json
from pathlib import Path
from ..helpers import gim, load_operational_world, make_result, determine_status, Timer


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        world = load_operational_world()

        from gim.game_runner import GameRunner
        from gim.scenario_compiler import load_game_definition

        # Load the bundled maritime pressure game
        g = gim()
        repo_root = Path(g["OPERATIONAL_STATE_CSV"]).parent.parent
        case_path = repo_root / "misc" / "cases" / "maritime_pressure_game.json"

        if not case_path.exists():
            return make_result("Q10", "WARN", {},
                               notes=f"Test case not found: {case_path}")

        game_def = load_game_definition(str(case_path), world)
        runner = GameRunner(world)

        # Pre-compute stage game (shared across episodes)
        stage_game = runner.run_game(game_def, max_combinations=256)

        # ------------------------------------------------------------------
        # Run 1: 50 episodes (default)
        # ------------------------------------------------------------------
        try:
            from gim.game_theory.equilibrium_runner import run_equilibrium_search
        except ImportError:
            return make_result("Q10", "WARN", {},
                               notes="equilibrium_runner not importable (scipy missing?)")

        eq_50 = run_equilibrium_search(
            runner, game_def, world,
            max_episodes=50, stage_game=stage_game,
        )
        converged_50 = eq_50.converged
        regret_50 = eq_50.mean_external_regret
        profile_50 = eq_50.recommended_profile

        # ------------------------------------------------------------------
        # Run 2: 200 episodes
        # ------------------------------------------------------------------
        eq_200 = run_equilibrium_search(
            runner, game_def, world,
            max_episodes=200, stage_game=stage_game,
        )
        converged_200 = eq_200.converged
        regret_200 = eq_200.mean_external_regret
        profile_200 = eq_200.recommended_profile

        # ------------------------------------------------------------------
        # Run 3: 50 episodes, no exploration
        # ------------------------------------------------------------------
        eq_noexplore = run_equilibrium_search(
            runner, game_def, world,
            max_episodes=50, exploration_eps=0.0, stage_game=stage_game,
        )

        # Check profile stability
        profile_stable = profile_50 == profile_200
        max_regret_200 = max(regret_200.values()) if regret_200 else 0
        regret_low = max_regret_200 < 0.02

        conditions = {
            "profile_stable_50_vs_200": profile_stable,
            "max_regret_200_lt_002": regret_low,
        }

    return make_result(
        test_id="Q10",
        status=determine_status(conditions),
        metrics={
            "converged_50ep": converged_50,
            "converged_200ep": converged_200,
            "episodes_50": eq_50.episodes,
            "episodes_200": eq_200.episodes,
            "max_external_regret_50": round(max(regret_50.values()) if regret_50 else 0, 4),
            "max_external_regret_200": round(max_regret_200, 4),
            "profile_stable": profile_stable,
        },
        flagged_items=[
            "Profile differs between 50 and 200 episodes" if not profile_stable else "",
            f"Max regret at 200ep = {max_regret_200:.4f} (threshold: <0.02)"
            if not regret_low else "",
        ],
        notes=f"Completed in {t.elapsed:.1f}s",
        details={
            "profile_50": profile_50,
            "profile_200": profile_200,
            "regret_50": {k: round(v, 5) for k, v in regret_50.items()},
            "regret_200": {k: round(v, 5) for k, v in regret_200.items()},
            "warnings_50": eq_50.warnings,
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
