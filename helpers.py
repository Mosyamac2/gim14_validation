"""
Common helper functions for GIM-14 validation tests.

Provides world loading, simulation running, parameter perturbation,
and result formatting utilities shared across all test modules.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure the GIM-14 package is importable.  The environment variable
# GIM14_REPO must point to the root of the GIM-14 repository, or it falls
# back to ../GIM-GIM14 relative to this project.
# ---------------------------------------------------------------------------
_REPO_ROOT_CANDIDATES = [
    os.environ.get("GIM14_REPO", ""),
    str(Path(__file__).resolve().parent.parent / "GIM-GIM14"),
    str(Path(__file__).resolve().parent.parent / "GIM_14"),
]

GIM_REPO_ROOT: Optional[Path] = None
for _candidate in _REPO_ROOT_CANDIDATES:
    _p = Path(_candidate)
    if _p.is_dir() and (_p / "gim").is_dir():
        GIM_REPO_ROOT = _p
        break

if GIM_REPO_ROOT is not None and str(GIM_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(GIM_REPO_ROOT))


# ---------------------------------------------------------------------------
# GIM-14 imports (lazy, so import errors are reported at call-time)
# ---------------------------------------------------------------------------

def _import_gim():
    """Import GIM-14 core modules and return them as a namespace dict."""
    from gim.core.world_factory import make_world_from_csv
    from gim.core.simulation import step_world
    from gim.core.policy import make_policy_map, simple_rule_based_policy
    from gim.core.core import WorldState, AgentState, RelationState
    from gim.paths import DEFAULT_STATE_CSV, OPERATIONAL_STATE_CSV
    import gim.core.calibration_params as cal
    return {
        "make_world_from_csv": make_world_from_csv,
        "step_world": step_world,
        "make_policy_map": make_policy_map,
        "simple_rule_based_policy": simple_rule_based_policy,
        "WorldState": WorldState,
        "AgentState": AgentState,
        "RelationState": RelationState,
        "DEFAULT_STATE_CSV": DEFAULT_STATE_CSV,
        "OPERATIONAL_STATE_CSV": OPERATIONAL_STATE_CSV,
        "cal": cal,
    }


# Cache the import so we only do it once
_gim_cache: Optional[dict] = None


def gim():
    """Return the cached GIM-14 module namespace."""
    global _gim_cache
    if _gim_cache is None:
        _gim_cache = _import_gim()
    return _gim_cache


# ---------------------------------------------------------------------------
# World loading helpers
# ---------------------------------------------------------------------------

def load_operational_world(max_agents: int | None = None):
    """Load the 57-actor operational world state."""
    g = gim()
    csv_path = str(g["OPERATIONAL_STATE_CSV"])
    if not Path(csv_path).exists():
        csv_path = str(g["DEFAULT_STATE_CSV"])
    return g["make_world_from_csv"](csv_path, max_agents=max_agents)


def load_compact_world(max_agents: int | None = None):
    """Load the 20-actor compact world state."""
    g = gim()
    return g["make_world_from_csv"](str(g["DEFAULT_STATE_CSV"]), max_agents=max_agents)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def run_steps(world, years: int = 5, seed: int = 42,
              enable_extreme_events: bool = False,
              policy_mode: str = "simple") -> list:
    """
    Run *years* simulation steps and return the trajectory
    [world_t0, world_t1, ..., world_tN].
    """
    g = gim()
    os.environ["SIM_SEED"] = str(seed)
    if not enable_extreme_events:
        os.environ["DISABLE_EXTREME_EVENTS"] = "1"
    else:
        os.environ.pop("DISABLE_EXTREME_EVENTS", None)
    os.environ["NO_LLM"] = "1"

    policies = g["make_policy_map"](world.agents.keys(), mode=policy_mode)
    memory: dict = {}
    w = copy.deepcopy(world)
    trajectory = [copy.deepcopy(w)]
    for _ in range(years):
        w = g["step_world"](w, policies, enable_extreme_events=enable_extreme_events)
        trajectory.append(copy.deepcopy(w))
    return trajectory


# ---------------------------------------------------------------------------
# Parameter perturbation helpers
# ---------------------------------------------------------------------------

class ParamPatch:
    """Context manager to temporarily monkey-patch a calibration parameter."""

    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value
        self._original: float | None = None

    def __enter__(self):
        cal = gim()["cal"]
        self._original = getattr(cal, self.name)
        setattr(cal, self.name, self.value)
        return self

    def __exit__(self, *exc):
        cal = gim()["cal"]
        setattr(cal, self.name, self._original)


def patch_param(name: str, value: float) -> ParamPatch:
    """Return a context manager that temporarily sets a calibration param."""
    return ParamPatch(name, value)


# ---------------------------------------------------------------------------
# Agent state perturbation
# ---------------------------------------------------------------------------

def perturb_agent_field(world, agent_id: str, field_path: str, factor: float):
    """
    Multiply a nested agent field by *factor*.
    field_path examples: 'economy.gdp', 'society.trust_gov'
    Returns the deepcopy of the world with the perturbation applied.
    """
    w = copy.deepcopy(world)
    agent = w.agents[agent_id]
    parts = field_path.split(".")
    obj = agent
    for part in parts[:-1]:
        obj = getattr(obj, part)
    current = getattr(obj, parts[-1])
    setattr(obj, parts[-1], current * factor)
    return w


def set_agent_field(world, agent_id: str, field_path: str, value: float):
    """Set a nested agent field to an absolute value."""
    w = copy.deepcopy(world)
    agent = w.agents[agent_id]
    parts = field_path.split(".")
    obj = agent
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
    return w


# ---------------------------------------------------------------------------
# Relation helpers
# ---------------------------------------------------------------------------

def get_relation(world, from_id: str, to_id: str):
    """Retrieve the relation state from from_id -> to_id."""
    return world.relations.get(from_id, {}).get(to_id)


def set_relation_field(world, from_id: str, to_id: str, field: str, value: float):
    """Set a relation field in a deepcopy of the world."""
    w = copy.deepcopy(world)
    rel = w.relations[from_id][to_id]
    setattr(rel, field, value)
    return w


# ---------------------------------------------------------------------------
# Metric extraction helpers
# ---------------------------------------------------------------------------

def agent_gdp(world, agent_id: str) -> float:
    return world.agents[agent_id].economy.gdp


def global_gdp(world) -> float:
    return sum(a.economy.gdp for a in world.agents.values())


def global_avg_trust(world) -> float:
    agents = list(world.agents.values())
    return sum(a.society.trust_gov for a in agents) / len(agents)


def global_avg_tension(world) -> float:
    agents = list(world.agents.values())
    return sum(a.society.social_tension for a in agents) / len(agents)


def global_co2(world) -> float:
    return world.global_state.co2


def count_distressed_agents(world) -> int:
    return sum(
        1 for a in world.agents.values()
        if a.credit_zone in ("distressed", "default")
    )


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def make_result(test_id: str, status: str, metrics: dict,
                flagged_items: list | None = None,
                notes: str = "",
                details: dict | None = None) -> dict:
    """Build a standardized test result dictionary."""
    result = {
        "test_id": test_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "metrics": metrics,
        "flagged_items": flagged_items or [],
        "notes": notes,
    }
    if details:
        result["details"] = details
    return result


def determine_status(conditions: dict[str, bool]) -> str:
    """
    Given a dict of condition_name -> passed (bool),
    return PASS / WARN / FAIL.
    """
    if all(conditions.values()):
        return "PASS"
    failing = sum(1 for v in conditions.values() if not v)
    if failing <= len(conditions) // 3:
        return "WARN"
    return "FAIL"


# ---------------------------------------------------------------------------
# Timer utility
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
