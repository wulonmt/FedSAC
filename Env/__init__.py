from gymnasium.envs.registration import register

register(
    id='CartPoleSwingUpFixInitState-v1',
    entry_point='Env.envs:CartPoleSwingUpFixInitStateV1',
    max_episode_steps=500,
)

register(
    id='CartPoleSwingUpFixInitState-v2',
    entry_point='Env.envs:CartPoleSwingUpFixPosGoalOriented',
    max_episode_steps=800,
)

register(
    id='MountainCarFixPos-v0',
    entry_point='Env.envs:MountainCarFixPos',
    max_episode_steps=300,
    reward_threshold=-110.0,
)

register(
    id='MountainCarFixPos-v1',
    entry_point='Env.envs:MountainCarFixPosGoalOriented',
    max_episode_steps=300,
    reward_threshold=-110.0,
)

register(
    id='PendulumFixPos-v0',
    entry_point='Env.envs:PendulumFixPos',
    max_episode_steps=200,
)

register(
    id='PendulumFixPos-v1',
    entry_point='Env.envs:PendulumFixPosGoalOriented',
    max_episode_steps=200,
)

register(
    id='HopperFixLength-v0',
    entry_point='Env.envs:HopperFixLength',
    max_episode_steps=500,
    reward_threshold=3800.0,
)

register(
    id='HalfCheetahFixLength-v0',
    entry_point='Env.envs:HalfCheetahFixLength',
    max_episode_steps=300,
    reward_threshold=4800.0,
)

register(
    id='CrowdedHighway-v0',
    entry_point='Env.envs:CrowdedHighway',
    max_episode_steps=500,
)

register(
    id='CrowdedHighway-v1',
    entry_point='Env.envs:CrowdedHighway_V1',
    max_episode_steps=500,
)

register(
    id='CarRacingFixSeed-v0',
    entry_point='Env.envs:CarRacingFixSeed',
)