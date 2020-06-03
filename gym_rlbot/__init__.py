from gym.envs.registration import register

register(
    id='rlbot-v0',
    entry_point='gym_rlbot.envs:RlbotEnv',
)