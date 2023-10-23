"""
Vizualize Lunar Lander Trajectory
author: Aishik Pyne (@aishik-pyne)
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import gymnasium as gym
from optimalcontrol.state import State, Action, Trajectory, render_trajectory
from gymnasium.envs.box2d.lunar_lander import LunarLander

def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def render_lander_motion(state: State, trajectory: Trajectory, info: dict):
    # Matplot lib create new plot
    if plt.fignum_exists(200):
        fig = plt.figure(200)
    else:
        # Figure is closed
        fig = plt.figure(200)
    
    # Draw horizontal line segment
    goal_x1, goal_y1, goal_x2, goal_y2 = info["goal"]

    # Clear the current figure
    plt.cla()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    # plot_arrow(state.lander_x, state.lander_y, state.lander_theta)
    # plt.plot(cx, cy, "-r", label="course")

    plt.plot([goal_x1, goal_x2], [0, 0], "-g", label="goal")
    
    # Goal boundary
    plt.axvline(x=goal_x1, color="g", linestyle="--")
    plt.axvline(x=goal_x2, color="g", linestyle="--")

    # Game boundary
    plt.axvline(x=-1, color="r", linestyle="--")
    plt.axvline(x=1, color="r", linestyle="--")

    plt.plot(trajectory.lander_x, trajectory.lander_y, "-b", label="trajectory")
    # plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")

    plt.grid(True)
    # Set y lim
    plt.ylim(-1, 2)
    plt.xlim(-2, 2)
    plt.title("Speed[km/h]:" + str(state.vel * 3.6)[:4])
    plt.pause(0.001)



if __name__ == "__main__":
    
    from optimalcontrol.utils.telemetry import Telemetry
    telemetry = Telemetry()

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


    trajectory = Trajectory()
    observation, info = env.reset()
    helipad_x1_cor, helipad_y_cor, helipad_x2_cor, helipad_y_cor = info["goal"]
    helipad_x_center_cor = (helipad_x1_cor + helipad_x2_cor) / 2

    for t in range(5000):    
        state = State(observation)

        # Policy
        # action = env.action_space.sample() # [0, 0]
        curr_state = State(observation)

        action = [0.01, (curr_state.lander_x-helipad_x_center_cor)]

        next_observation, reward, terminated, truncated, info = env.step(action)


        trajectory.append(State(observation),
                            Action(action),
                            State(next_observation), t)
        if t % 5 == 0: 
            telemetry.update(trajectory)
        # render_lander_motion(state, trajectory, info)
        # render_trajectory(trajectory)
        if terminated or truncated:
            # render_trajectory(trajectory)
            break
            trajectory = States()
            next_observation, info = env.reset()
            helipad_x1_cor, helipad_y_cor, helipad_x2_cor, helipad_y_cor = info["goal"]
        
        observation = next_observation


    render_trajectory(trajectory)
    # render_lander_motion(state, trajectory, info)
    plt.show()