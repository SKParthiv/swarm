from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from swarm.env.simple_spread_v0 import parallel_env
from swarm.policies import BlindGreedyPolicy, DynamicAverageConsensusPolicy, OracleAssignerPolicy, PolicyContext
from swarm.throttling import ObservationThrottleConfig, estimate_effective_bps


@dataclass
class EnvironmentConfig:
    max_cycles: int = 150
    continuous_actions: bool = False


@dataclass
class SwarmConfig:
    n_agents: int = 3
    n_landmarks: int = 3


@dataclass
class ThrottleConfig:
    keep_velocity: bool = False
    quantization_bits: int = 8
    value_range: float = 2.0
    update_interval: int = 5


@dataclass
class ConsensusConfig:
    gain: float = 0.45
    neighbor_radius: float = 0.8
    repulsion_gain: float = 0.35


@dataclass
class ExperimentConfig:
    episodes: int = 5
    seed: int = 7


@dataclass
class Level1Config:
    environment: EnvironmentConfig
    swarm: SwarmConfig
    throttle: ThrottleConfig
    consensus: ConsensusConfig
    experiment: ExperimentConfig



def _section(data: dict[str, Any], key: str, cls: type):
    return cls(**data.get(key, {}))



def load_config(path: str | Path) -> Level1Config:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    return Level1Config(
        environment=_section(raw, "environment", EnvironmentConfig),
        swarm=_section(raw, "swarm", SwarmConfig),
        throttle=_section(raw, "throttle", ThrottleConfig),
        consensus=_section(raw, "consensus", ConsensusConfig),
        experiment=_section(raw, "experiment", ExperimentConfig),
    )



def build_policy(name: str, context: PolicyContext):
    if name == "blind":
        return BlindGreedyPolicy(context)
    if name == "oracle":
        return OracleAssignerPolicy(context)
    if name == "consensus":
        return DynamicAverageConsensusPolicy(context)
    raise ValueError(f"Unknown policy: {name}")



def run_episode(env, policy, max_steps: int, seed: int):
    observations, _ = env.reset(seed=seed)
    policy.reset()

    total_reward = 0.0
    step = 0

    while env.agents and step < max_steps:
        actions = policy.act(observations, list(env.agents), step)
        observations, rewards, terminations, truncations, _ = env.step(actions)

        total_reward += sum(rewards.values())
        step += 1

        if all(terminations.values()) or all(truncations.values()):
            break

    return {"steps": step, "team_reward": total_reward}



def main() -> None:
    parser = argparse.ArgumentParser(description="Run Level 1 simple_spread_v3 baseline policies")
    parser.add_argument("--config", default="configs/level1_simple_spread.yaml")
    parser.add_argument("--policy", choices=["blind", "oracle", "consensus"], default="consensus")
    args = parser.parse_args()

    cfg = load_config(args.config)
    throttle_cfg = ObservationThrottleConfig(**asdict(cfg.throttle))
    env = parallel_env(
        n_agents=cfg.swarm.n_agents,
        max_cycles=cfg.environment.max_cycles,
        continuous_actions=cfg.environment.continuous_actions,
        throttle_config=throttle_cfg,
    )

    policy_context = PolicyContext(
        n_agents=cfg.swarm.n_agents,
        n_landmarks=cfg.swarm.n_landmarks,
        neighbor_radius=cfg.consensus.neighbor_radius,
        consensus_gain=cfg.consensus.gain,
        repulsion_gain=cfg.consensus.repulsion_gain,
        comm_update_interval=cfg.throttle.update_interval,
    )
    policy = build_policy(args.policy, policy_context)

    initial_obs, _ = env.reset(seed=cfg.experiment.seed)
    obs_dim = len(next(iter(initial_obs.values())))
    bps = estimate_effective_bps(
        obs_dim=obs_dim,
        n_agents=cfg.swarm.n_agents,
        quantization_bits=cfg.throttle.quantization_bits,
        update_interval=cfg.throttle.update_interval,
        keep_velocity=cfg.throttle.keep_velocity,
    )

    print(f"Policy: {args.policy}")
    print(f"Estimated effective shared bitrate: {bps:.1f} bits/s")

    metrics = []
    for i in range(cfg.experiment.episodes):
        seed = cfg.experiment.seed + i
        result = run_episode(env, policy, cfg.environment.max_cycles, seed=seed)
        metrics.append(result)
        print(
            f"Episode {i + 1}: steps={result['steps']}, "
            f"team_reward={result['team_reward']:.3f}, seed={seed}"
        )

    avg_steps = sum(metric["steps"] for metric in metrics) / len(metrics)
    avg_reward = sum(metric["team_reward"] for metric in metrics) / len(metrics)
    print(f"Average: steps={avg_steps:.2f}, team_reward={avg_reward:.3f}")

    env.close()


if __name__ == "__main__":
    main()
