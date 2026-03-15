"""
Q12 — LLM Fallback Rate and Silent Degradation

Measure how often LLM policy calls fail and silently fall back to
simple_rule_based_policy, and quantify the GDP impact on fallback agents.
"""
from __future__ import annotations
import copy
import functools
import os
from ..helpers import (
    gim, load_compact_world, require_api_key,
    make_result, determine_status, Timer,
)


class LLMCallTracker:
    """Wraps llm_policy to track calls, successes, and fallbacks."""

    def __init__(self):
        self.total_calls = 0
        self.successes = 0
        self.fallbacks = 0
        self.fallback_agents: dict[str, int] = {}  # agent_id -> count
        self.errors: list[dict] = []

    def wrap(self, original_llm_policy, original_simple_policy):
        """Return a patched llm_policy function that counts outcomes."""
        tracker = self

        def tracked_policy(obs, memory_summary=None):
            tracker.total_calls += 1
            agent_id = obs.agent_id
            try:
                if memory_summary is not None:
                    action = original_llm_policy(obs, memory_summary)
                else:
                    action = original_llm_policy(obs)
                # Check if the action is the simple fallback
                # The real llm_policy catches exceptions internally and
                # returns simple_rule_based_policy on failure, printing
                # "LLM policy error for ...".  We detect this via the
                # explanation field.
                if action.explanation == "baseline do-nothing policy":
                    tracker.fallbacks += 1
                    tracker.fallback_agents[agent_id] = (
                        tracker.fallback_agents.get(agent_id, 0) + 1
                    )
                else:
                    tracker.successes += 1
                return action
            except Exception as exc:
                tracker.fallbacks += 1
                tracker.fallback_agents[agent_id] = (
                    tracker.fallback_agents.get(agent_id, 0) + 1
                )
                tracker.errors.append({
                    "agent_id": agent_id,
                    "error": str(exc),
                    "type": type(exc).__name__,
                })
                return original_simple_policy(obs)

        # Tag so step_world recognizes it as an async/LLM policy
        tracked_policy.__gim_async_policy__ = False  # run sync for tracking clarity
        return tracked_policy


def run(*, verbose: bool = False) -> dict:
    with Timer() as t:
        require_api_key()
        world = load_compact_world()
        g = gim()

        os.environ["SIM_SEED"] = "42"
        os.environ.pop("NO_LLM", None)
        os.environ.pop("USE_SIMPLE_POLICIES", None)
        os.environ["POLICY_MODE"] = "llm"
        os.environ["DISABLE_EXTREME_EVENTS"] = "1"
        os.environ.setdefault("LLM_MAX_CONCURRENCY", "4")

        from gim.core.policy import llm_policy, simple_rule_based_policy
        from gim.core.simulation import step_world
        from gim.core.observation import build_observation
        from gim.core.memory import summarize_agent_memory

        tracker = LLMCallTracker()
        tracked_fn = tracker.wrap(llm_policy, simple_rule_based_policy)

        # Build policy map pointing all agents to tracked function
        policies = {aid: tracked_fn for aid in world.agents}

        w = copy.deepcopy(world)
        memory: dict = {}
        trajectory = [copy.deepcopy(w)]
        for _ in range(3):
            w = step_world(
                w, policies, memory=memory,
                enable_extreme_events=False,
            )
            trajectory.append(copy.deepcopy(w))

        fallback_rate = tracker.fallbacks / max(tracker.total_calls, 1)

        # Compare GDP growth for fallback vs non-fallback agents
        fallback_ids = set(tracker.fallback_agents.keys())
        non_fallback_ids = set(world.agents.keys()) - fallback_ids

        def avg_gdp_growth(ids, traj):
            if not ids:
                return 0.0
            growths = []
            for aid in ids:
                g0 = traj[0].agents[aid].economy.gdp
                g3 = traj[3].agents[aid].economy.gdp
                growths.append((g3 - g0) / max(g0, 1e-9))
            return sum(growths) / len(growths)

        fb_growth = avg_gdp_growth(fallback_ids, trajectory)
        nonfb_growth = avg_gdp_growth(non_fallback_ids, trajectory)
        gdp_gap = nonfb_growth - fb_growth

        # Max fallbacks per agent
        max_per_agent = max(tracker.fallback_agents.values()) if tracker.fallback_agents else 0

        conditions = {
            "fallback_rate_lt_010": fallback_rate < 0.10,
            "no_agent_gt_1_fallback": max_per_agent <= 1,
        }

    return make_result(
        test_id="Q12",
        status=determine_status(conditions),
        metrics={
            "total_llm_calls": tracker.total_calls,
            "successes": tracker.successes,
            "fallbacks": tracker.fallbacks,
            "fallback_rate": round(fallback_rate, 4),
            "fallback_agent_count": len(fallback_ids),
            "max_fallbacks_per_agent": max_per_agent,
            "avg_gdp_growth_fallback": round(fb_growth, 4) if fallback_ids else None,
            "avg_gdp_growth_nonfallback": round(nonfb_growth, 4) if non_fallback_ids else None,
            "gdp_growth_gap": round(gdp_gap, 4) if fallback_ids else None,
        },
        flagged_items=[
            f"Fallback rate = {fallback_rate:.1%} ({tracker.fallbacks}/{tracker.total_calls})"
            if fallback_rate >= 0.10 else "",
            f"Agent with {max_per_agent} fallbacks in 3 years"
            if max_per_agent > 1 else "",
        ],
        notes=f"Completed in {t.elapsed:.1f}s. Errors: {len(tracker.errors)}.",
        details={
            "fallback_agents": dict(tracker.fallback_agents),
            "errors": tracker.errors[:10],
            "conditions": {k: bool(v) for k, v in conditions.items()},
        },
    )
