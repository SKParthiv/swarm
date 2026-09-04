from __future__ import annotations

import numpy as np

from swarm.policies import BasePolicy, discrete_action_from_vector, neighbors_within_radius, parse_observation, steering_vector


class DynamicAverageConsensusPolicy(BasePolicy):
    def __init__(self, context):
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
            _, self_pos, landmark_rel, other_rel, ordered_others = parse_observation(
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
            neighbors = neighbors_within_radius(other_rel, ordered_others, self.context.neighbor_radius)
            signals = [self.last_broadcast[agent]]
            for neighbor in neighbors:
                if neighbor in self.last_broadcast:
                    signals.append(self.last_broadcast[neighbor])
            mixed = np.mean(np.stack(signals, axis=0), axis=0)

            innovation = local_utilities[agent] - self.prev_u[agent]
            self.z[agent] = (1.0 - gain) * self.z[agent] + gain * mixed + innovation
            self.prev_u[agent] = local_utilities[agent].copy()

            target_idx = int(np.argmax(self.z[agent]))
            target_abs = self_pos + landmark_rel[target_idx]
            desired = steering_vector(self_pos, target_abs, other_rel, self.context)
            actions[agent] = discrete_action_from_vector(desired)

        return actions
