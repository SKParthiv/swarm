from __future__ import annotations

import numpy as np

from swarm.policies import BasePolicy, discrete_action_from_vector, nearest_landmark_index, parse_observation, steering_vector


class BlindGreedyPolicy(BasePolicy):
    def act(self, observations: dict[str, np.ndarray], agents: list[str], step: int) -> dict[str, int]:
        actions: dict[str, int] = {}
        for agent in agents:
            _, self_pos, landmark_rel, other_rel, _ = parse_observation(
                observations[agent], self.context.n_landmarks, self.context.n_agents
            )
            target_idx = nearest_landmark_index(landmark_rel)
            target_abs = self_pos + landmark_rel[target_idx]
            desired = steering_vector(self_pos, target_abs, other_rel, self.context)
            actions[agent] = discrete_action_from_vector(desired)
        return actions
