"""Shared fixtures for the Phase 1 (Level 1) test suite.

Defaults mirror configs/level1_simple_spread.yaml so tests exercise the
same configuration the README's Quick Start runs use.
"""

from __future__ import annotations

import pytest

from swarm.env.simple_spread_v0 import parallel_env
from swarm.policies import PolicyContext
from swarm.throttling import ObservationThrottleConfig


@pytest.fixture(scope="session")
def default_context() -> PolicyContext:
    """Policy context matching configs/level1_simple_spread.yaml."""
    return PolicyContext(
        n_agents=3,
        n_landmarks=3,
        neighbor_radius=0.8,
        consensus_gain=0.45,
        repulsion_gain=0.35,
        comm_update_interval=5,
    )


@pytest.fixture
def throttle_config() -> ObservationThrottleConfig:
    """Throttle config matching configs/level1_simple_spread.yaml."""
    return ObservationThrottleConfig(
        keep_velocity=False,
        quantization_bits=8,
        value_range=2.0,
        update_interval=5,
    )


def make_env(max_cycles: int = 25, throttle_config=None, n_agents: int = 3):
    """Build a fresh parallel env (optionally throttled) for a single test."""
    return parallel_env(
        n_agents=n_agents,
        max_cycles=max_cycles,
        throttle_config=throttle_config,
    )
