import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander

env_options = {
    "discreteaction-deterministic-fo": {
        "continuous": False,
        "stochastic": False,
        "enable_wind": True,
        "wind_power": 0,
        "turbulence_power": 0,
        "partially_observable": False,
    },
    "discreteaction-stochastic-fo": {
        "continuous": False,
        "stochastic": True,
        "enable_wind": True,
        "wind_power": 15,
        "turbulence_power": 2,
        "partially_observable": False,
    },
    "continuous_action-stochastic-fo": {
        "continuous": "continuous",
        "stochastic": True,
        "enable_wind": True,
        "wind_power": 15,
        "turbulence_power": 2,
        "partially_observable": False,
    },
}

def get_train_env(env_option, seed=42):
    option = env_options[env_option]

    env: LunarLander = gym.make(
        "LunarLander-v2",
        continuous = option['continuous'],
        gravity = -10.0,
        enable_wind = option['stochastic'],
        wind_power = option['wind_power'],
        turbulence_power = option['turbulence_power'],
        partially_observable = option['partially_observable'],
        render_mode=None
    )
    env._max_episode_steps = 300
    env.reset(seed=seed)

    return env

def get_eval_env(env_option, seed=24):
    option = env_options[env_option]

    env: LunarLander = gym.make(
        "LunarLander-v2",
        continuous = option['continuous'],
        gravity = -10.0,
        enable_wind = option['stochastic'],
        wind_power = option['wind_power'],
        turbulence_power = option['turbulence_power'],
        partially_observable = option['partially_observable'],
        render_mode="human"
    )
    env._max_episode_steps = 300
    env.reset(seed=seed)

    return env