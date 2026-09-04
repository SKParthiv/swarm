from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return Level1Config(
        environment=_section(raw, "environment", EnvironmentConfig),
        swarm=_section(raw, "swarm", SwarmConfig),
        throttle=_section(raw, "throttle", ThrottleConfig),
        consensus=_section(raw, "consensus", ConsensusConfig),
        experiment=_section(raw, "experiment", ExperimentConfig),
    )
