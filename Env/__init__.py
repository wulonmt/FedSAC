from gymnasium.envs.registration import register

register(
    id='CartPoleSwingUpFixInitState-v1',
    entry_point='Env.envs:CartPoleSwingUpFixInitStateV1',
    max_episode_steps=1000,
)

register(
    id='MountainCarFixPos-v0',
    entry_point='Env.envs:MountainCarFixPos',
    max_episode_steps=300,
    reward_threshold=-110.0,
)

register(
    id='PendulumFixPos-v0',
    entry_point='Env.envs:PendulumFixPos',
    max_episode_steps=200,
)
