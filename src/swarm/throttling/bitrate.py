from __future__ import annotations


def estimate_effective_bps(
    obs_dim: int,
    n_agents: int,
    quantization_bits: int,
    update_interval: int,
    base_hz: float = 10.0,
    dropped_velocity_dims: int = 2,
    keep_velocity: bool = False,
) -> float:
    effective_dim = obs_dim if keep_velocity else max(0, obs_dim - dropped_velocity_dims)
    hz = base_hz / max(1, update_interval)
    return float(effective_dim * quantization_bits * n_agents * hz)
