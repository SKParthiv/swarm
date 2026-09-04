from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np


@dataclass
class PolicyContext:
    n_agents: int
    n_landmarks: int
    neighbor_radius: float = 0.8
    consensus_gain: float = 0.45
    repulsion_gain: float = 0.35
    comm_update_interval: int = 5


class BasePolicy:
    def __init__(self, context: PolicyContext):
        self.context = context

    def reset(self):
        return None

    def act(self, observations: dict[str, np.ndarray], agents: list[str], step: int) -> dict[str, int]:
        raise NotImplementedError



def parse_observation(
    obs: np.ndarray,
    n_landmarks: int,
    n_agents: int,
    agent_order: list[str] | None = None,
    current_agent: str | None = None,
):
    obs = np.asarray(obs, dtype=np.float32)
    self_vel = obs[0:2]
    self_pos = obs[2:4]

    lm_start = 4
    lm_end = lm_start + (2 * n_landmarks)
    landmarks_rel = obs[lm_start:lm_end].reshape(n_landmarks, 2)

    others_start = lm_end
    others_end = others_start + max(0, 2 * (n_agents - 1))
    other_agents_rel = obs[others_start:others_end]
    if other_agents_rel.size:
        other_agents_rel = other_agents_rel.reshape(-1, 2)
    else:
        other_agents_rel = np.zeros((0, 2), dtype=np.float32)

    ordered_others: list[str] = []
    if agent_order and current_agent and len(agent_order) == n_agents:
        ordered_others = [a for a in agent_order if a != current_agent]

    return self_vel, self_pos, landmarks_rel, other_agents_rel, ordered_others



def nearest_landmark_index(landmarks_rel: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(landmarks_rel, axis=1)))



def optimal_assignment(agent_pos: np.ndarray, landmark_pos: np.ndarray) -> list[int]:
    n_agents = agent_pos.shape[0]
    if n_agents > 8:
        raise ValueError(
            f"optimal_assignment uses factorial search; n_agents={n_agents} is too large. "
            "Use a polynomial-time assignment algorithm (e.g., Hungarian) or reduce n_agents."
        )

    indices = range(landmark_pos.shape[0])
    best = None
    best_cost = float("inf")

    for perm in permutations(indices, n_agents):
        cost = 0.0
        for i, lm_idx in enumerate(perm):
            cost += float(np.linalg.norm(agent_pos[i] - landmark_pos[lm_idx]))
        if cost < best_cost:
            best_cost = cost
            best = list(perm)

    return best if best is not None else list(range(n_agents))



def neighbors_within_radius(rel_positions: np.ndarray, labels: list[str], radius: float) -> list[str]:
    neighbors: list[str] = []
    for i, rel in enumerate(rel_positions):
        if i >= len(labels):
            break
        if float(np.linalg.norm(rel)) <= radius:
            neighbors.append(labels[i])
    return neighbors



def steering_vector(
    self_pos: np.ndarray,
    target_abs: np.ndarray,
    other_rel: np.ndarray,
    context: PolicyContext,
) -> np.ndarray:
    attraction = target_abs - self_pos
    attraction_norm = np.linalg.norm(attraction)
    if attraction_norm > 1e-8:
        attraction = attraction / attraction_norm

    repulsion = np.zeros(2, dtype=np.float32)
    for rel in other_rel:
        dist = float(np.linalg.norm(rel))
        if 1e-8 < dist < context.neighbor_radius:
            repulsion -= (rel / dist) * ((context.neighbor_radius - dist) / context.neighbor_radius)

    return attraction + context.repulsion_gain * repulsion



def discrete_action_from_vector(vector: np.ndarray) -> int:
    vx, vy = float(vector[0]), float(vector[1])
    if abs(vx) < 0.05 and abs(vy) < 0.05:
        return 0
    if abs(vx) >= abs(vy):
        return 2 if vx > 0 else 1
    return 4 if vy > 0 else 3


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swarm.policies.blind_greedy import BlindGreedyPolicy
    from swarm.policies.dynamic_average_consensus import DynamicAverageConsensusPolicy
    from swarm.policies.oracle_assigner import OracleAssignerPolicy

__all__ = [
    "BasePolicy",
    "PolicyContext",
    "BlindGreedyPolicy",
    "OracleAssignerPolicy",
    "DynamicAverageConsensusPolicy",
]


def __getattr__(name: str):
    if name == "BlindGreedyPolicy":
        from swarm.policies.blind_greedy import BlindGreedyPolicy as _BlindGreedyPolicy

        return _BlindGreedyPolicy
    if name == "DynamicAverageConsensusPolicy":
        from swarm.policies.dynamic_average_consensus import (
            DynamicAverageConsensusPolicy as _DynamicAverageConsensusPolicy,
        )

        return _DynamicAverageConsensusPolicy
    if name == "OracleAssignerPolicy":
        from swarm.policies.oracle_assigner import OracleAssignerPolicy as _OracleAssignerPolicy

        return _OracleAssignerPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
