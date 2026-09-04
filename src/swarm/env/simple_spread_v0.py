from __future__ import annotations

from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils.conversions import parallel_to_aec

from swarm.throttling.observation_filters import ObservationThrottleConfig, ThrottledObservationAdapter



def raw_parallel_env(
    n_agents: int = 3,
    max_cycles: int = 150,
    continuous_actions: bool = False,
):
    return simple_spread_v3.parallel_env(
        N=n_agents,
        max_cycles=max_cycles,
        continuous_actions=continuous_actions,
    )



def parallel_env(
    n_agents: int = 3,
    max_cycles: int = 150,
    continuous_actions: bool = False,
    throttle_config: ObservationThrottleConfig | None = None,
):
    env = raw_parallel_env(
        n_agents=n_agents,
        max_cycles=max_cycles,
        continuous_actions=continuous_actions,
    )
    if throttle_config is not None:
        env = ThrottledObservationAdapter(env, throttle_config)
    return env



def env(
    n_agents: int = 3,
    max_cycles: int = 150,
    continuous_actions: bool = False,
    throttle_config: ObservationThrottleConfig | None = None,
):
    return parallel_to_aec(
        parallel_env(
            n_agents=n_agents,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            throttle_config=throttle_config,
        )
    )
