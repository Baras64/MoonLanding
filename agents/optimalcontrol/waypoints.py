import numpy as np
from state import State, Trajectory

def find_waypoints(init_state: State, goal_pos, num_waypoints=5):
    """Finds waypoints for the lunar lander to follow.
    """
    # Define one way point for each height    
    x = np.linspace(init_state.lander_x, goal_pos[0], num_waypoints)
    y = np.linspace(init_state.lander_y/2, goal_pos[1],  num_waypoints)
    theta = np.zeros(num_waypoints)
    vel_x = np.zeros(num_waypoints)
    vel_y = np.zeros(num_waypoints)
    vel_theta = np.zeros(num_waypoints)

    # Define waypoints
    waypoints = np.stack((x, y, theta, vel_x, vel_y, vel_theta), axis=1)

    return waypoints

if __name__ == "__main__":
    
    from gymnasium.envs.box2d.lunar_lander import LunarLander
    import gymnasium as gym
    from state import State
    from matplotlib import pyplot as plt

    
    env: LunarLander = gym.make(
        "LunarLander-v2",
        continuous = True,
        gravity = -10.0,
        enable_wind = False,
        wind_power = 0,
        turbulence_power = 0,
        render_mode="human", # Turn off for training
        randomize_start = True,
        ref_lander = True
    )

    observation, info = env.reset()
    helipad_x1_cor, helipad_y_cor, helipad_x2_cor, helipad_y_cor = info["goal"]
    goal_x, goal_y = (helipad_x1_cor + helipad_x2_cor) / 2, helipad_y_cor

    waypoints = find_waypoints(State(observation), (goal_x, goal_y))

    plt.plot([helipad_x1_cor, helipad_x2_cor], [helipad_y_cor, helipad_y_cor], "-g", label="goal")
    plt.plot(waypoints[:, 0], waypoints[:, 1], "xg", label="waypoints")
    # Goal boundary
    plt.axvline(x=helipad_x1_cor, color="g", linestyle="--")
    plt.axvline(x=helipad_x2_cor, color="g", linestyle="--")

    # Game boundary
    plt.axvline(x=-1, color="r", linestyle="--")
    plt.axvline(x=1, color="r", linestyle="--")

    # plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")

    plt.grid(True)
    # Set y lim
    plt.ylim(-1, 2)
    plt.xlim(-2, 2)
    plt.title("Waypoints")
    plt.show()