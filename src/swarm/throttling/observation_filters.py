from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ObservationThrottleConfig:
    keep_velocity: bool = False
    quantization_bits: int = 8
    value_range: float = 2.0
    update_interval: int = 5


class ThrottledObservationAdapter:
    """Applies dimensionality, quantization, and temporal throttling to observations."""

    def __init__(self, env, config: ObservationThrottleConfig):
        self.env = env
        self.config = config
        self._step = 0
        self._cache: dict[str, np.ndarray] = {}

    @property
    def agents(self):
        return self.env.agents

    def reset(self, *args, **kwargs):
        self._step = 0
        self._cache = {}
        observations, info = self.env.reset(*args, **kwargs)
        return self._process(observations), info

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self._step += 1
        return self._process(observations), rewards, terminations, truncations, infos

    def close(self):
        self.env.close()

    def __getattr__(self, name: str):
        return getattr(self.env, name)
    def _process(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        processed: dict[str, np.ndarray] = {}
        refresh = self._step % max(1, self.config.update_interval) == 0

        for agent, obs in observations.items():
            transformed = np.array(obs, dtype=np.float32, copy=True)

            if not self.config.keep_velocity and transformed.size >= 2:
                transformed[0:2] = 0.0

            transformed = self._quantize(transformed)

            if refresh or agent not in self._cache:
                self._cache[agent] = transformed

            processed[agent] = np.array(self._cache[agent], copy=True)

        return processed

    def _quantize(self, vector: np.ndarray) -> np.ndarray:
        bits = int(max(1, self.config.quantization_bits))
        levels = (2**bits) - 1
        if levels <= 1:
            return np.zeros_like(vector)

        clipped = np.clip(vector, -self.config.value_range, self.config.value_range)
        normalized = (clipped + self.config.value_range) / (2 * self.config.value_range)
        bucketed = np.round(normalized * levels) / levels
        return (bucketed * (2 * self.config.value_range) - self.config.value_range).astype(np.float32)
