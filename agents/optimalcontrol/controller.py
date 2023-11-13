"""
Vizualize Lunar Lander Trajectory
author: Aishik Pyne (@aishik-pyne)
"""
import sys
sys.path.append('..')

import numpy as np
import math
import matplotlib.pyplot as plt
import gymnasium as gym
from optimalcontrol.state import State, Action, Trajectory, render_trajectory
from optimalcontrol.utils.telemetry import Telemetry
from optimalcontrol.waypoints import find_waypoints
from gymnasium.envs.box2d.lunar_lander import LunarLander
from simple_pid import PID
import argparse
import logging

# Setup Console logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Lunar Lander PID Controller')
    parser.add_argument('--Kp', type=float, default=0.1, help='Proportional gain')
    parser.add_argument('--Ki', type=float, default=0.01, help='Integral gain')
    parser.add_argument('--Kd', type=float, default=0.005, help='Derivative gain')

    return parser.parse_args()

class LunarPIDController:
    
        def __init__(self, Kp_v=1, Ki_v=0.1, Kd_v=0, Kp_w=1, Ki_w=0.1, Kd_w=0):
            self.Kp_v = Kp_v
            self.Ki_v = Ki_v
            self.Kd_v = Kd_v
            
            self.Kp_w = Kp_w
            self.Ki_w = Ki_w
            self.Kd_w = Kd_w

            self._waypoint_idx = 0
            self.waypoints = None
    
        def set_watpoints(self, waypoints):
            self._waypoint_idx = 0
            self.waypoints = waypoints

            self._update_pids()

        def _update_pids(self):

            self.altitude_pid = PID(self.Kp*0.1, self.Ki, self.Kd, 
                             setpoint=self.waypoints[self._waypoint_idx][0: 2])
            self.w_pid = PID(self.Kp*2, self.Ki, self.Kd*10, 
                             setpoint=self.waypoints[self._waypoint_idx][2])
            
            # self.x_pid = PID(self.Kp, self.Ki, self.Kd, \
            #                  setpoint=self.waypoints[self._waypoint_idx][0])
            # self.y_pid = PID(self.Kp, self.Ki, self.Kd, \
            #                  setpoint=self.waypoints[self._waypoint_idx][1])
            # self.theta_pid = PID(self.Kp, self.Ki, self.Kd, \
            #                      setpoint=self.waypoints[self._waypoint_idx][2])
            # self.vel_x_pid = PID(self.Kp, self.Ki, self.Kd, \
            #                      setpoint=self.waypoints[self._waypoint_idx][3])
            # self.vel_y_pid = PID(self.Kp, self.Ki, self.Kd, \
            #                      setpoint=self.waypoints[self._waypoint_idx][4])
            # self.vel_theta_pid = PID(self.Kp, self.Ki, self.Kd, \
            #                          setpoint=self.waypoints[self._waypoint_idx][5])

        def _next_waypoint(self):
            logger.info("Reached waypoint %d", self._waypoint_idx)
            if self._waypoint_idx < len(self.waypoints):
                self.waypoint = self.waypoints[self._waypoint_idx]
                self._waypoint_idx += 1
            else:
                self.waypoint = self.waypoints[-1]
            
            self._update_pids()

        def __call__(self, state):
            if self.waypoints is None:
                raise RuntimeError("No trajectory specified!")
            
            # Check if state is close to waypoint
            if np.linalg.norm(state - self.waypoints[self._waypoint_idx]) < 0.1:
                self._next_waypoint()

            # Compute control
            # x_control = self.x_pid(state[0])
            # y_control = self.y_pid(state[1])
            # theta_control = self.theta_pid(state[2])
            # vel_x_control = self.vel_x_pid(state[3])
            # vel_y_control = self.vel_y_pid(state[4])
            # vel_theta_control = self.vel_theta_pid(state[5])

            # v = y_control
            # w = theta_control

            error_v = np.linalg.norm(state[0:2] - self.waypoints[self._waypoint_idx][0:2])
            error_w = state[2] - self.waypoints[self._waypoint_idx][2]

            w = -self.w_pid(state[2])
            v = np.linalg.norm(self.altitude_pid(state[0:2]))

            # Log control
            logger.info("Control: v=%.2f, w=%.2f \t|\t Error: v=%.2f, w=%.2f", v, w, error_v, error_w)
            
            return np.array([v, w])


if __name__ == "__main__":

    args = parse_args()
    
    # telemetry = Telemetry()

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

    # Extract goal state
    helipad_x1_cor, helipad_y_cor, helipad_x2_cor, helipad_y_cor = info["goal"]
    goal_x, goal_y = (helipad_x1_cor + helipad_x2_cor) / 2, 0

    # Find waypoints
    waypoints = find_waypoints(init_state=State(observation), goal_pos=[goal_x, goal_y], num_waypoints=5)

    # Initialize controller
    controller = LunarPIDController(args.Kp, args.Ki, args.Kd)
    controller.set_watpoints(waypoints)

    while True:
        # Policy
        curr_state = State(observation)
        action = controller(curr_state.vector)

        next_observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

        observation = next_observation