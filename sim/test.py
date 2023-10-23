import gymnasium as gym

env = gym.make(
    "LunarLander-v2",
    continuous = False,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power = 1.5,
    render_mode="human",
    randomize_start = True,
    ref_lander = True
)

env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(observation)

    if terminated or truncated:
        observation, info = env.reset()

env.close()