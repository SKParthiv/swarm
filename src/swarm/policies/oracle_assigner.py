from __future__ import annotations

import numpy as np

from swarm.policies import BasePolicy, discrete_action_from_vector, optimal_assignment, parse_observation, steering_vector


class OracleAssignerPolicy(BasePolicy):
    def act(self, observations: dict[str, np.ndarray], agents: list[str], step: int) -> dict[str, int]:
        agent_positions = []
        landmark_estimates = []

        for agent in agents:
            _, self_pos, landmark_rel, _, _ = parse_observation(
                observations[agent], self.context.n_landmarks, self.context.n_agents
            )
            agent_positions.append(self_pos)
            landmark_estimates.append(self_pos + landmark_rel)

        agent_positions_np = np.vstack(agent_positions)
        landmark_positions = np.mean(np.stack(landmark_estimates, axis=0), axis=0)
        assignment = optimal_assignment(agent_positions_np, landmark_positions)

        actions: dict[str, int] = {}
        for i, agent in enumerate(agents):
            _, self_pos, _, other_rel, _ = parse_observation(
                observations[agent], self.context.n_landmarks, self.context.n_agents
            )
            target_abs = landmark_positions[assignment[i]]
            desired = steering_vector(self_pos, target_abs, other_rel, self.context)
            actions[agent] = discrete_action_from_vector(desired)
        return actions
