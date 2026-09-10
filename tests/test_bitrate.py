"""Tests for effective-bitrate estimation I(t) (compressed bits per second)."""

from __future__ import annotations

import pytest

from swarm.throttling import estimate_effective_bps


class TestEstimateEffectiveBps:
    def test_exact_formula(self):
        # effective_dim * bits * n_agents * (base_hz / update_interval)
        bps = estimate_effective_bps(
            obs_dim=18, n_agents=3, quantization_bits=8, update_interval=5,
            base_hz=10.0, dropped_velocity_dims=2, keep_velocity=False,
        )
        assert bps == pytest.approx(16 * 8 * 3 * 2.0)

    def test_velocity_dims_counted_when_kept(self):
        without = estimate_effective_bps(obs_dim=18, n_agents=3, quantization_bits=8, update_interval=5)
        with_vel = estimate_effective_bps(
            obs_dim=18, n_agents=3, quantization_bits=8, update_interval=5, keep_velocity=True
        )
        assert with_vel == pytest.approx(without * 18 / 16)

    def test_zero_bits_clamped_to_one(self):
        bps = estimate_effective_bps(obs_dim=18, n_agents=3, quantization_bits=0, update_interval=5)
        assert bps == pytest.approx(16 * 1 * 3 * 2.0)

    def test_update_interval_clamped_to_one(self):
        bps = estimate_effective_bps(obs_dim=18, n_agents=3, quantization_bits=8, update_interval=0)
        assert bps == pytest.approx(16 * 8 * 3 * 10.0)

    def test_bitrate_scales_inversely_with_interval(self):
        fast = estimate_effective_bps(obs_dim=18, n_agents=3, quantization_bits=8, update_interval=1)
        slow = estimate_effective_bps(obs_dim=18, n_agents=3, quantization_bits=8, update_interval=10)
        assert fast == pytest.approx(10.0 * slow)

    def test_bitrate_scales_linearly_with_quantization(self):
        coarse = estimate_effective_bps(obs_dim=18, n_agents=3, quantization_bits=4, update_interval=5)
        fine = estimate_effective_bps(obs_dim=18, n_agents=3, quantization_bits=8, update_interval=5)
        assert fine == pytest.approx(2.0 * coarse)

    def test_returns_float(self):
        assert isinstance(
            estimate_effective_bps(obs_dim=18, n_agents=3, quantization_bits=8, update_interval=5),
            float,
        )

    def test_zero_obs_dim_without_velocity(self):
        bps = estimate_effective_bps(obs_dim=2, n_agents=3, quantization_bits=8, update_interval=5)
        assert bps == 0.0

    @pytest.mark.parametrize(
        "obs_dim,n_agents,bits,interval,keep_velocity,expected",
        [
            (18, 3, 8, 5, False, 768.0),
            (18, 3, 8, 5, True, 864.0),
            (10, 2, 4, 2, False, 320.0),
            (6, 1, 1, 1, False, 40.0),
        ],
    )
    def test_parametrated_reference_values(self, obs_dim, n_agents, bits, interval, keep_velocity, expected):
        bps = estimate_effective_bps(
            obs_dim=obs_dim, n_agents=n_agents, quantization_bits=bits,
            update_interval=interval, keep_velocity=keep_velocity,
        )
        assert bps == pytest.approx(expected)
