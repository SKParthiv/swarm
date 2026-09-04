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


class BlindGreedyPolicy(BasePolicy):
    def act(self, observations: dict[str, np.ndarray], agents: list[str], step: int) -> dict[str, int]:
        actions: dict[str, int] = {}
        for agent in agents:
            _, self_pos, landmark_rel, other_rel, _ = _parse_observation(
                observations[agent], self.context.n_landmarks, self.context.n_agents
            )
            target_idx = _nearest_landmark_index(landmark_rel)
            target_abs = self_pos + landmark_rel[target_idx]
            desired = _steering_vector(self_pos, target_abs, other_rel, self.context)
            actions[agent] = _discrete_action_from_vector(desired)
        return actions


class OracleAssignerPolicy(BasePolicy):
    def act(self, observations: dict[str, np.ndarray], agents: list[str], step: int) -> dict[str, int]:
        agent_positions = []
        landmark_estimates = []

        for agent in agents:
            _, self_pos, landmark_rel, _, _ = _parse_observation(
                observations[agent], self.context.n_landmarks, self.context.n_agents
            )
            agent_positions.append(self_pos)
            landmark_estimates.append(self_pos + landmark_rel)

        agent_positions_np = np.vstack(agent_positions)
        landmark_positions = np.mean(np.stack(landmark_estimates, axis=0), axis=0)
        assignment = _optimal_assignment(agent_positions_np, landmark_positions)

        actions: dict[str, int] = {}
        for i, agent in enumerate(agents):
            _, self_pos, _, other_rel, _ = _parse_observation(
                observations[agent], self.context.n_landmarks, self.context.n_agents
            )
            target_abs = landmark_positions[assignment[i]]
            desired = _steering_vector(self_pos, target_abs, other_rel, self.context)
            actions[agent] = _discrete_action_from_vector(desired)
        return actions


class DynamicAverageConsensusPolicy(BasePolicy):
    def __init__(self, context: PolicyContext):
        super().__init__(context)
        self.z: dict[str, np.ndarray] = {}
        self.prev_u: dict[str, np.ndarray] = {}
        self.last_broadcast: dict[str, np.ndarray] = {}

    def reset(self):
        self.z.clear()
        self.prev_u.clear()
        self.last_broadcast.clear()

    def act(self, observations: dict[str, np.ndarray], agents: list[str], step: int) -> dict[str, int]:
        agent_order = list(agents)
        refreshed = step % max(1, self.context.comm_update_interval) == 0

        local_utilities: dict[str, np.ndarray] = {}
        parsed = {}
        for agent in agent_order:
            _, self_pos, landmark_rel, other_rel, ordered_others = _parse_observation(
                observations[agent], self.context.n_landmarks, self.context.n_agents, agent_order, agent
            )
            distances = np.linalg.norm(landmark_rel, axis=1)
            utility = 1.0 / (distances + 1e-6)
            utility = utility / (np.sum(utility) + 1e-9)
            local_utilities[agent] = utility
            parsed[agent] = (self_pos, landmark_rel, other_rel, ordered_others)

            if agent not in self.z:
                self.z[agent] = utility.copy()
                self.prev_u[agent] = utility.copy()
                self.last_broadcast[agent] = utility.copy()

        if refreshed:
            for agent in agent_order:
                self.last_broadcast[agent] = local_utilities[agent].copy()

        actions: dict[str, int] = {}
        gain = self.context.consensus_gain

        for agent in agent_order:
            self_pos, landmark_rel, other_rel, ordered_others = parsed[agent]
            neighbors = _neighbors_within_radius(other_rel, ordered_others, self.context.neighbor_radius)
            signals = [self.last_broadcast[agent]]
            for n in neighbors:
                if n in self.last_broadcast:
                    signals.append(self.last_broadcast[n])
            mixed = np.mean(np.stack(signals, axis=0), axis=0)

            innovation = local_utilities[agent] - self.prev_u[agent]
            self.z[agent] = (1.0 - gain) * self.z[agent] + gain * mixed + innovation
            self.prev_u[agent] = local_utilities[agent].copy()

            target_idx = int(np.argmax(self.z[agent]))
            target_abs = self_pos + landmark_rel[target_idx]
            desired = _steering_vector(self_pos, target_abs, other_rel, self.context)
            actions[agent] = _discrete_action_from_vector(desired)

        return actions



def _parse_observation(
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



def _nearest_landmark_index(landmarks_rel: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(landmarks_rel, axis=1)))



def _optimal_assignment(agent_pos: np.ndarray, landmark_pos: np.ndarray) -> list[int]:
    n_agents = agent_pos.shape[0]
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



def _neighbors_within_radius(rel_positions: np.ndarray, labels: list[str], radius: float) -> list[str]:
    neighbors: list[str] = []
    for i, rel in enumerate(rel_positions):
        if i >= len(labels):
            break
        if float(np.linalg.norm(rel)) <= radius:
            neighbors.append(labels[i])
    return neighbors



def _steering_vector(
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



def _discrete_action_from_vector(vector: np.ndarray) -> int:
    vx, vy = float(vector[0]), float(vector[1])
    if abs(vx) < 0.05 and abs(vy) < 0.05:
        return 0
    if abs(vx) >= abs(vy):
        return 2 if vx > 0 else 1
    return 4 if vy > 0 else 3
