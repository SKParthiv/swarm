"""Tests for observation throttling: dimensionality, quantization, temporal sparsity.

Covers the Phase 1 information-throttling axes from the README:
dimensionality truncation, resolution quantization, and frequency throttling.
"""

from __future__ import annotations

import numpy as np
import pytest

from swarm.throttling import ObservationThrottleConfig, ThrottledObservationAdapter

from tests.conftest import make_env


class TestObservationThrottleConfig:
    def test_defaults(self):
        cfg = ObservationThrottleConfig()
        assert cfg.keep_velocity is False
        assert cfg.quantization_bits == 8
        assert cfg.value_range == 2.0
        assert cfg.update_interval == 5


class TestDimensionalityTruncation:
    def test_velocity_dims_zeroed_by_default(self, throttle_config):
        env = make_env(throttle_config=throttle_config)
        try:
            obs, _ = env.reset(seed=3)
            for agent, vector in obs.items():
                assert np.all(vector[0:2] == 0.0), (
                    "velocity dims (0:2) must be truncated when keep_velocity=False"
                )
        finally:
            env.close()

    def test_velocity_dims_kept_when_configured(self, throttle_config):
        throttle_config.keep_velocity = True
        env = make_env(throttle_config=throttle_config)
        try:
            obs, _ = env.reset(seed=3)
            for agent, vector in obs.items():
                assert not np.allclose(vector[0:2], 0.0), (
                    "velocity dims must survive when keep_velocity=True"
                )
        finally:
            env.close()

    def test_observation_shape_preserved(self, throttle_config):
        env = make_env(throttle_config=throttle_config)
        try:
            obs, _ = env.reset(seed=3)
            for agent, vector in obs.items():
                assert vector.shape == (18,), f"unexpected obs shape for {agent}"
        finally:
            env.close()


class TestResolutionQuantization:
    def test_values_lie_on_quantization_grid(self, throttle_config):
        env = make_env(throttle_config=throttle_config)
        try:
            obs, _ = env.reset(seed=3)
            levels = (2 ** throttle_config.quantization_bits) - 1
            span = 2 * throttle_config.value_range
            for agent, vector in obs.items():
                for value in np.asarray(vector, dtype=np.float64).flatten():
                    if value == 0.0:
                        continue  # truncated dims
                    scaled = ((value + throttle_config.value_range) / span) * levels
                    assert np.isclose(scaled, round(scaled), atol=1e-4), (
                        f"value {value} off the {throttle_config.quantization_bits}-bit grid"
                    )
        finally:
            env.close()

    def test_quantized_values_within_value_range(self, throttle_config):
        env = make_env(throttle_config=throttle_config)
        try:
            obs, _ = env.reset(seed=3)
            for agent, vector in obs.items():
                assert np.all(np.abs(vector) <= throttle_config.value_range + 1e-6)
        finally:
            env.close()

    def test_quantize_known_grid(self):
        cfg = ObservationThrottleConfig(keep_velocity=True, quantization_bits=2, value_range=1.0, update_interval=1)
        adapter = ThrottledObservationAdapter(_NullEnv(), cfg)
        out = adapter._quantize(np.array([0.0, 0.33, -0.9], dtype=np.float32))
        # 2-bit grid over [-1, 1]: levels=3 -> grid {-1, -1/3, 1/3, 1}
        assert np.allclose(out, [1 / 3, 1 / 3, -1.0], atol=1e-6)

    def test_single_bit_quantizes_to_zeros(self):
        cfg = ObservationThrottleConfig(quantization_bits=1)
        adapter = ThrottledObservationAdapter(_NullEnv(), cfg)
        out = adapter._quantize(np.array([1.5, -2.0, 0.7], dtype=np.float32))
        assert np.all(out == 0.0)

    def test_clipping_to_value_range(self):
        cfg = ObservationThrottleConfig(keep_velocity=True, quantization_bits=8, value_range=1.0, update_interval=1)
        adapter = ThrottledObservationAdapter(_NullEnv(), cfg)
        out = adapter._quantize(np.array([5.0, -5.0], dtype=np.float32))
        assert np.allclose(out, [1.0, -1.0], atol=1e-6)


class _NullEnv:
    """Minimal stand-in so the adapter can be constructed for unit tests."""

    agents: list[str] = []

    def close(self):
        pass


class TestTemporalSparsity:
    def test_stale_observation_repeated_between_refreshes(self, throttle_config):
        env = make_env(throttle_config=throttle_config)
        try:
            obs, _ = env.reset(seed=3)
            actions = {a: 0 for a in env.agents}
            snapshots = [obs]
            for _ in range(throttle_config.update_interval + 1):
                obs, *_ = env.step(actions)
                snapshots.append(obs)
            # Between refresh steps the cached observation must be returned unchanged.
            for prev, cur in zip(snapshots, snapshots[1:]):
                for agent in cur:
                    assert np.array_equal(cur[agent], prev[agent]), (
                        "cached observation must be byte-identical between refreshes"
                    )
        finally:
            env.close()

    def test_refresh_happens_at_interval_boundary(self, throttle_config):
        env = make_env(throttle_config=throttle_config)
        try:
            obs, _ = env.reset(seed=3)
            actions = {a: 2 for a in env.agents}
            before = None
            changed_at = None
            for step in range(1, 2 * throttle_config.update_interval + 2):
                obs, *_ = env.step(actions)
                if before is not None and changed_at is None:
                    if any(not np.array_equal(obs[a], before[a]) for a in obs):
                        changed_at = step
                before = obs
            assert changed_at is not None, "observation must eventually refresh"
            assert changed_at % throttle_config.update_interval == 0, (
                f"refresh at step {changed_at} not aligned to interval "
                f"{throttle_config.update_interval}"
            )
        finally:
            env.close()


class TestAdapterPlumbing:
    def test_agents_property_delegates(self, throttle_config):
        env = make_env(throttle_config=throttle_config)
        try:
            env.reset(seed=3)
            assert env.agents == env.env.agents
        finally:
            env.close()

    def test_getattr_falls_through_to_inner_env(self, throttle_config):
        env = make_env(throttle_config=throttle_config)
        try:
            # `possible_agents` exists on the raw MPE env, not the adapter class
            assert hasattr(env, "possible_agents")
        finally:
            env.close()

    def test_reset_is_deterministic_per_seed(self, throttle_config):
        env = make_env(throttle_config=throttle_config)
        try:
            obs1, _ = env.reset(seed=3)
            env.step({a: 0 for a in env.agents})
            obs2, _ = env.reset(seed=3)
            for agent in obs2:
                assert np.array_equal(obs1[agent], obs2[agent]), (
                    "same seed after reset must reproduce the same processed obs"
                )
        finally:
            env.close()

    def test_step_returns_five_tuple(self, throttle_config):
        env = make_env(throttle_config=throttle_config)
        try:
            env.reset(seed=3)
            result = env.step({a: 0 for a in env.agents})
            assert len(result) == 5
            observations, rewards, terminations, truncations, infos = result
            assert isinstance(observations, dict)
            assert isinstance(rewards, dict)
            assert isinstance(terminations, dict)
            assert isinstance(truncations, dict)
        finally:
            env.close()

    def test_processed_observations_are_copies(self, throttle_config):
        """Mutating a returned observation must not corrupt the adapter cache."""
        env = make_env(throttle_config=throttle_config)
        try:
            obs, _ = env.reset(seed=3)
            agent = next(iter(obs))
            obs[agent][:] = 999.0
            obs2, *_ = env.step({a: 0 for a in env.agents})
            assert not np.all(obs2[agent] == 999.0)
        finally:
            env.close()
