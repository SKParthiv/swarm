"""Tests for the Level 1 runner: config loading, policy construction,
episode execution, and the blind/oracle anchor ordering that the
README's P(t) normalization depends on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swarm.experiments.level1_runner import (
    ExperimentConfig,
    build_policy,
    load_config,
    run_episode,
)
from swarm.policies import (
    BlindGreedyPolicy,
    DynamicAverageConsensusPolicy,
    OracleAssignerPolicy,
    PolicyContext,
)
from swarm.throttling import ObservationThrottleConfig

from tests.conftest import make_env

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "level1_simple_spread.yaml"


class TestLoadConfig:
    def test_loads_repo_config(self):
        cfg = load_config(CONFIG_PATH)
        assert cfg.environment.max_cycles == 150
        assert cfg.environment.continuous_actions is False
        assert cfg.swarm.n_agents == 3
        assert cfg.swarm.n_landmarks == 3
        assert cfg.throttle.quantization_bits == 8
        assert cfg.throttle.update_interval == 5
        assert cfg.consensus.gain == 0.45
        assert cfg.experiment.episodes == 5
        assert cfg.experiment.seed == 7

    def test_empty_config_uses_defaults(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.touch()
        cfg = load_config(config_file)
        assert cfg.environment.max_cycles == 150
        assert cfg.swarm.n_agents == 3
        assert cfg.experiment.episodes == 5

    def test_partial_config_overrides_only_given_sections(self, tmp_path):
        config_file = tmp_path / "partial.yaml"
        config_file.write_text("swarm:\n  n_agents: 5\n")
        cfg = load_config(config_file)
        assert cfg.swarm.n_agents == 5
        assert cfg.throttle.quantization_bits == 8  # untouched default


class TestBuildPolicy:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("blind", BlindGreedyPolicy),
            ("oracle", OracleAssignerPolicy),
            ("consensus", DynamicAverageConsensusPolicy),
        ],
    )
    def test_builds_expected_policy(self, default_context, name, cls):
        assert isinstance(build_policy(name, default_context), cls)

    def test_unknown_policy_raises(self, default_context):
        with pytest.raises(ValueError, match="Unknown policy"):
            build_policy("bogus", default_context)


class TestRunEpisode:
    def test_returns_steps_and_team_reward(self, default_context, throttle_config):
        env = make_env(throttle_config=throttle_config, max_cycles=25)
        try:
            result = run_episode(env, BlindGreedyPolicy(default_context), max_steps=25, seed=3)
            assert set(result) == {"steps", "team_reward"}
            assert 0 < result["steps"] <= 25
            assert isinstance(result["team_reward"], float)
        finally:
            env.close()

    def test_deterministic_per_seed(self, default_context, throttle_config):
        env = make_env(throttle_config=throttle_config, max_cycles=25)
        try:
            policy = BlindGreedyPolicy(default_context)
            r1 = run_episode(env, policy, max_steps=25, seed=3)
            r2 = run_episode(env, policy, max_steps=25, seed=3)
            assert r1 == r2, "same seed must reproduce identical episode results"
        finally:
            env.close()

    def test_respects_max_steps(self, default_context, throttle_config):
        env = make_env(throttle_config=throttle_config, max_cycles=50)
        try:
            result = run_episode(
                env, BlindGreedyPolicy(default_context), max_steps=10, seed=3
            )
            assert result["steps"] <= 10
        finally:
            env.close()


class TestBaselineAnchors:
    """The README's P(t) normalization assumes oracle >= blind on average.
    These are statistical smoke tests over a small seed ensemble, not
    strict guarantees: they assert the anchor ordering holds on the
    default configuration, which is what Phase 1 validation needs."""

    N_SEEDS = 5
    MAX_CYCLES = 60

    def _mean_reward(self, policy, throttle_config):
        rewards = []
        env = make_env(throttle_config=throttle_config, max_cycles=self.MAX_CYCLES)
        try:
            for seed in range(self.N_SEEDS):
                result = run_episode(env, policy, max_steps=self.MAX_CYCLES, seed=seed)
                rewards.append(result["team_reward"])
        finally:
            env.close()
        return float(np.mean(rewards))

    def test_oracle_at_least_blind(self, default_context, throttle_config):
        blind = self._mean_reward(BlindGreedyPolicy(default_context), throttle_config)
        oracle = self._mean_reward(OracleAssignerPolicy(default_context), throttle_config)
        assert oracle >= blind, (
            f"oracle anchor ({oracle:.3f}) below blind anchor ({blind:.3f}); "
            "P(t) normalization assumption violated"
        )

    def test_consensus_runs_full_episode(self, default_context, throttle_config):
        env = make_env(throttle_config=throttle_config, max_cycles=self.MAX_CYCLES)
        try:
            policy = DynamicAverageConsensusPolicy(default_context)
            for seed in range(self.N_SEEDS):
                result = run_episode(env, policy, max_steps=self.MAX_CYCLES, seed=seed)
                assert result["steps"] > 0
                assert np.isfinite(result["team_reward"])
        finally:
            env.close()


class TestEndToEndRunner:
    def test_runner_main_smoke(self, capsys, tmp_path):
        """The Quick Start command must run end-to-end without error."""
        import subprocess
        import sys

        repo_root = Path(__file__).resolve().parents[1]
        # Keep the smoke test fast: a tiny config with 2 episodes.
        smoke_config = tmp_path / "smoke.yaml"
        smoke_config.write_text(
            "environment:\n  max_cycles: 30\n"
            "swarm:\n  n_agents: 3\n  n_landmarks: 3\n"
            "throttle:\n  keep_velocity: false\n  quantization_bits: 8\n"
            "  value_range: 2.0\n  update_interval: 5\n"
            "consensus:\n  gain: 0.45\n  neighbor_radius: 0.8\n  repulsion_gain: 0.35\n"
            "experiment:\n  episodes: 2\n  seed: 7\n"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "swarm.experiments.level1_runner",
                "--config",
                str(smoke_config),
                "--policy",
                "consensus",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"runner failed:\n{result.stderr}"
        output = capsys.readouterr()
        assert "Policy: consensus" in result.stdout
        assert "Estimated effective shared bitrate" in result.stdout
        assert "Average:" in result.stdout
