"""Tests for policy primitives and the three Phase 1 policies:
blind greedy (P=0 anchor), oracle assigner (P=1 anchor), and
dynamic average consensus (the throttled driving policy).
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from swarm.policies import (
    BasePolicy,
    BlindGreedyPolicy,
    DynamicAverageConsensusPolicy,
    OracleAssignerPolicy,
    discrete_action_from_vector,
    nearest_landmark_index,
    neighbors_within_radius,
    optimal_assignment,
    parse_observation,
    steering_vector,
)
from tests.conftest import make_env


def make_observation(
    self_pos=(0.0, 0.0),
    landmark_rel=((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)),
    other_rel=((0.5, 0.0), (0.0, -0.5)),
    n_landmarks=3,
):
    obs = np.zeros(4 + 2 * n_landmarks + 2 * len(other_rel), dtype=np.float32)
    obs[2:4] = self_pos
    obs[4 : 4 + 2 * n_landmarks] = np.asarray(landmark_rel, dtype=np.float32).flatten()
    obs[4 + 2 * n_landmarks :] = np.asarray(other_rel, dtype=np.float32).flatten()
    return obs


class TestParseObservation:
    def test_layout_split(self):
        obs = make_observation(
            self_pos=(0.3, -0.2),
            landmark_rel=((1, 0), (0, 1), (-1, 0)),
            other_rel=((0.5, 0.5), (-0.5, 0.5)),
        )
        self_vel, self_pos, landmarks, others, ordered = parse_observation(obs, 3, 3)
        assert np.allclose(self_pos, (0.3, -0.2))
        assert landmarks.shape == (3, 2)
        assert np.allclose(landmarks[0], (1, 0))
        assert others.shape == (2, 2)
        assert np.allclose(others[1], (-0.5, 0.5))
        assert ordered == []

    def test_ordered_others_respect_agent_order(self):
        obs = make_observation()
        _, _, _, _, ordered = parse_observation(
            obs, 3, 3, agent_order=["agent_0", "agent_1", "agent_2"], current_agent="agent_1"
        )
        assert ordered == ["agent_0", "agent_2"]

    def test_single_agent_has_empty_others(self):
        obs = np.zeros(4 + 2 * 3, dtype=np.float32)
        _, _, _, others, _ = parse_observation(obs, 3, 1)
        assert others.shape == (0, 2)


class TestNearestLandmarkIndex:
    def test_picks_closest(self):
        rel = np.array([[0.9, 0.0], [0.1, 0.1], [-5.0, 0.0]], dtype=np.float32)
        assert nearest_landmark_index(rel) == 1

    def test_tie_breaks_on_first(self):
        rel = np.array([[0.5, 0.0], [0.5, 0.0]], dtype=np.float32)
        assert nearest_landmark_index(rel) == 0


class TestOptimalAssignment:
    def test_perfect_assignment(self):
        agents = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
        landmarks = np.array([[10.0, 10.0], [0.0, 0.0], [5.0, 5.0]])
        assert optimal_assignment(agents, landmarks) == [1, 2, 0]

    def test_is_permutation(self):
        rng = np.random.default_rng(0)
        assignment = optimal_assignment(rng.normal(size=(4, 2)), rng.normal(size=(4, 2)))
        assert sorted(assignment) == [0, 1, 2, 3]

    def test_minimizes_total_cost(self):
        rng = np.random.default_rng(1)
        agents = rng.normal(size=(4, 2))
        landmarks = rng.normal(size=(4, 2))
        best = optimal_assignment(agents, landmarks)
        best_cost = sum(np.linalg.norm(agents[i] - landmarks[best[i]]) for i in range(4))
        for perm in permutations(range(4)):
            cost = sum(np.linalg.norm(agents[i] - landmarks[perm[i]]) for i in range(4))
            assert best_cost <= cost + 1e-9

    def test_rejects_too_many_agents(self):
        with pytest.raises(ValueError, match="n_agents=9"):
            optimal_assignment(np.zeros((9, 2)), np.zeros((9, 2)))


class TestNeighborsWithinRadius:
    def test_filters_by_radius(self):
        rel = np.array([[0.5, 0.0], [2.0, 0.0], [0.3, 0.4]], dtype=np.float32)
        assert neighbors_within_radius(rel, ["a", "b", "c"], radius=0.8) == ["a", "c"]

    def test_short_label_list_is_safe(self):
        rel = np.array([[0.1, 0.0], [0.1, 0.0]], dtype=np.float32)
        assert neighbors_within_radius(rel, ["a"], radius=0.8) == ["a"]


class TestSteeringVector:
    def test_attraction_normalized(self, default_context):
        v = steering_vector(np.zeros(2), np.array([10.0, 0.0]), np.zeros((0, 2)), default_context)
        assert np.allclose(v, (1.0, 0.0))

    def test_repulsion_pushes_away_from_close_neighbor(self, default_context):
        other = np.array([[0.4, 0.0]], dtype=np.float32)  # within radius 0.8
        v = steering_vector(np.zeros(2), np.array([10.0, 0.0]), other, default_context)
        assert v[0] < 1.0, "repulsion must reduce attraction toward a close neighbor"

    def test_no_repulsion_outside_radius(self, default_context):
        other = np.array([[2.0, 0.0]], dtype=np.float32)
        v = steering_vector(np.zeros(2), np.array([10.0, 0.0]), other, default_context)
        assert np.allclose(v, (1.0, 0.0))

    def test_zero_target_returns_zero(self, default_context):
        v = steering_vector(np.zeros(2), np.zeros(2), np.zeros((0, 2)), default_context)
        assert np.allclose(v, 0.0)


class TestDiscreteActionFromVector:
    @pytest.mark.parametrize(
        "vector,expected",
        [
            ((0.0, 0.0), 0),  # no-op
            ((0.01, 0.01), 0),  # below deadzone
            ((0.5, 0.1), 2),  # right
            ((-0.5, 0.1), 1),  # left
            ((0.1, 0.5), 4),  # up
            ((0.1, -0.5), 3),  # down
            ((0.5, 0.5), 2),  # tie -> horizontal wins
            ((-0.5, -0.5), 1),
        ],
    )
    def test_action_mapping(self, vector, expected):
        assert discrete_action_from_vector(np.array(vector, dtype=np.float32)) == expected


class TestBlindGreedyPolicy:
    def test_targets_nearest_landmark(self, default_context):
        policy = BlindGreedyPolicy(default_context)
        obs = {
            "agent_0": make_observation(landmark_rel=((0.2, 0.0), (1.5, 0.0), (2.0, 0.0))),
            "agent_1": make_observation(landmark_rel=((2.0, 0.0), (0.2, 0.0), (2.0, 0.0))),
        }
        actions = policy.act(obs, list(obs), step=0)
        assert actions["agent_0"] == 2  # nearest landmark is to the right
        assert actions["agent_1"] == 2  # nearest landmark (index 1) is also to the right

    def test_actions_are_valid_discrete(self, default_context):
        policy = BlindGreedyPolicy(default_context)
        actions = policy.act({"agent_0": make_observation()}, ["agent_0"], step=0)
        assert all(isinstance(a, int) and 0 <= a <= 4 for a in actions.values())


class TestOracleAssignerPolicy:
    def test_one_to_one_assignment_no_duplicates(self, default_context):
        """The oracle must produce a one-to-one landmark assignment (README: P=1.0 anchor)."""
        policy = OracleAssignerPolicy(default_context)
        obs = {
            "agent_0": make_observation(
                self_pos=(0.0, 0.0),
                landmark_rel=((1.0, 0.0), (0.0, 1.0), (-1.0, -1.0)),
                other_rel=((0.2, 0.0), (0.0, 0.2)),
            ),
            "agent_1": make_observation(
                self_pos=(0.2, 0.0),
                landmark_rel=((0.8, 0.0), (-0.2, 1.0), (-1.2, -1.0)),
                other_rel=((-0.2, 0.0), (0.0, 0.2)),
            ),
            "agent_2": make_observation(
                self_pos=(0.2, 0.2),
                landmark_rel=((0.8, -0.2), (-0.2, 0.8), (-1.2, -1.2)),
                other_rel=((-0.2, -0.2), (0.0, -0.2)),
            ),
        }
        # Landmark estimates are consistent across agents, so the oracle's
        # averaged estimates equal the true landmark positions.
        agent_positions = np.vstack([np.asarray(o[2:4]) for o in obs.values()])
        estimates = np.stack(
            [np.asarray(o[2:4]) + np.asarray(o[4:10]).reshape(3, 2) for o in obs.values()]
        )
        landmarks = np.mean(estimates, axis=0)
        assignment = optimal_assignment(agent_positions, landmarks)
        assert len(set(assignment)) == 3, f"duplicate landmark assignment: {assignment}"

        # The policy itself must run and emit valid discrete actions.
        actions = policy.act(obs, list(obs), step=0)
        assert all(isinstance(a, int) and 0 <= a <= 4 for a in actions.values())

    def test_actions_are_valid_discrete(self, default_context):
        policy = OracleAssignerPolicy(default_context)
        actions = policy.act({"agent_0": make_observation()}, ["agent_0"], step=0)
        assert all(isinstance(a, int) and 0 <= a <= 4 for a in actions.values())


class TestDynamicAverageConsensusPolicy:
    def test_reset_clears_state(self, default_context):
        policy = DynamicAverageConsensusPolicy(default_context)
        policy.act({"agent_0": make_observation()}, ["agent_0"], step=0)
        assert policy.z and policy.prev_u and policy.last_broadcast
        policy.reset()
        assert not policy.z and not policy.prev_u and not policy.last_broadcast

    def test_actions_are_valid_discrete(self, default_context, throttle_config):
        env = make_env(throttle_config=throttle_config, max_cycles=15)
        try:
            policy = DynamicAverageConsensusPolicy(default_context)
            obs, _ = env.reset(seed=3)
            policy.reset()
            for step in range(10):
                actions = policy.act(obs, list(env.agents), step)
                assert set(actions) == set(env.agents)
                assert all(isinstance(a, int) and 0 <= a <= 4 for a in actions.values())
                obs, *_ = env.step(actions)
        finally:
            env.close()

    def test_consensus_state_stays_normalized(self, default_context, throttle_config):
        """The consensus state z must remain a valid probability simplex."""
        env = make_env(throttle_config=throttle_config, max_cycles=15)
        try:
            policy = DynamicAverageConsensusPolicy(default_context)
            obs, _ = env.reset(seed=3)
            policy.reset()
            for step in range(20):
                actions = policy.act(obs, list(env.agents), step)
                obs, *_ = env.step(actions)
            for agent, z in policy.z.items():
                assert np.isclose(np.sum(z), 1.0, atol=1e-3), f"z for {agent} not normalized: {z}"
                assert np.all(z >= -1e-6), f"negative utility in z for {agent}: {z}"
        finally:
            env.close()

    def test_broadcast_refreshes_only_at_interval(self, default_context):
        policy = DynamicAverageConsensusPolicy(default_context)
        near = make_observation(landmark_rel=((0.2, 0.0), (1.0, 0.0), (1.0, 0.0)))
        far = make_observation(landmark_rel=((1.0, 0.0), (0.2, 0.0), (1.0, 0.0)))
        policy.act({"agent_0": near}, ["agent_0"], step=0)
        first_broadcast = policy.last_broadcast["agent_0"].copy()
        # steps 1..4 are not refresh steps (interval=5, refresh at step % 5 == 0)
        for step in range(1, 5):
            policy.act({"agent_0": far}, ["agent_0"], step=step)
        assert np.array_equal(policy.last_broadcast["agent_0"], first_broadcast), (
            "broadcast must be stale between refresh steps even as local utilities change"
        )
        # step 5 is a refresh step: the broadcast must track the new local utility
        policy.act({"agent_0": far}, ["agent_0"], step=5)
        assert not np.array_equal(policy.last_broadcast["agent_0"], first_broadcast)


class TestPolicyContract:
    @pytest.mark.parametrize(
        "make",
        [
            BlindGreedyPolicy,
            OracleAssignerPolicy,
            DynamicAverageConsensusPolicy,
        ],
    )
    def test_subclasses_share_base_and_reset(self, default_context, make):
        policy = make(default_context)
        assert isinstance(policy, BasePolicy)
        assert policy.reset() is None

    def test_base_policy_act_raises(self, default_context):
        with pytest.raises(NotImplementedError):
            BasePolicy(default_context).act({}, [], step=0)
