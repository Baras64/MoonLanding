#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:18:26 2023 

@author: Aishik Pyne

Usage:
    python evaluate.py --deterministic  --fully_observable 
    python evaluate.py --stochastic  --fully_observable 
    python evaluate.py --stochastic  --partially_observable 

"""
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm
import argparse

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Lunar Lander PID Controller')
    parser.add_argument('--Kp_alt', type=float, default=4.60245202, help='Proportional gain altitude')
    parser.add_argument('--Kd_alt', type=float, default=-3.57703678, help='Derivative gain altitude')
    parser.add_argument('--Kp_ang', type=float, default=-60.1764885, help='Proportional gain angle')
    parser.add_argument('--Kd_ang', type=float, default=40.0989124, help='Derivative gain angle')

    # Environemnt Flags
    parser.add_argument('--stochastic', action='store_true', help='Stochastic')
    parser.add_argument('--deterministic', dest='stochastic', action='store_false', help='Deterministic')
    parser.set_defaults(stochastic=False)

    parser.add_argument('--partially_observable', action='store_true', help='Partially Observable')
    parser.add_argument('--fully_observable', dest='partially_observable',  action='store_false', help='Fully Observable')
    parser.set_defaults(partially_observable=False)

    parser.add_argument('--render', action="store_true", help='Render')
    parser.set_defaults(render=False)

    # Run Flags
    parser.add_argument('--runs', type=int, default=10000, help='Number of runs')
    return parser.parse_args()


class Data():
    """tracks elements of the state"""
    def __init__(self):
        self.states = []
    
    def add(self,state):
        self.states.append(state)
        
    def graph(self):
        states = np.array(self.states).reshape(len(self.states),-1)
        plt.plot(states[:,0],label='x')
        plt.plot(states[:,1],label='y')
        plt.plot(states[:,2],label='vx')
        plt.plot(states[:,3],label='vy')
        plt.plot(states[:,4],label='theta')
        plt.plot(states[:,5],label='vtheta')
        plt.legend()
        plt.grid()
        plt.ylim(-1.1,1.1)
        plt.title('PID Control')
        plt.ylabel('Value')
        plt.xlabel('Steps')
        plt.show()
    

def pid(state, params):
    """ calculates settings based on pid control """
    # PID parameters
    kp_alt = params[0]  # proportional altitude
    kd_alt = params[1]  # derivative altitude
    kp_ang = params[2]  # proportional angle
    kd_ang = params[3]  # derivative angle

    x, y, vx, vy, theta, vtheta, left_leg, right_leg = state[:8]
    
    # Calculate setpoints (target values)
    alt_tgt = np.abs(x)
    ang_tgt = (.25*np.pi)*(x+vx)

    # Calculate error values
    alt_error = (alt_tgt - y)
    ang_error = (ang_tgt - theta)
    
    # Use PID to get adjustments
    alt_adj = kp_alt*alt_error + kd_alt*vy
    ang_adj = kp_ang*ang_error + kd_ang*vtheta
        
    # Gym wants them as np array (-1,1)
    a = np.array([alt_adj, ang_adj])
    a = np.clip(a, -1, +1)
    
    # If the legs are on the ground we made it, kill engines
    if(left_leg or right_leg):
        a[:] = 0   
    return a

def run(params, env, verbose=False):
    """ runs an episode given pid parameters """
    data = Data() 
    done = False
    state, info = env.reset()
    if verbose:
        env.render()
        # sleep(.005)
    data.add(state)
    total = 0
    while not done:
        a = pid(state,params)
        state, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        total += reward
        if verbose:
            env.render()
            # sleep(.005)
        data.add(state)
    return total, data

    
def main():
    args = parse_args()

    # Seed RNGs
    seed = 42
    np.random.seed(seed)

    # Parameters learned previously
    params = np.array([args.Kp_alt, args.Kd_alt, args.Kp_ang, args.Kd_ang])

    if args.stochastic:
        enable_wind = True
        wind_power = 15
        turbulence_power = 2
    else:
        enable_wind = False
        wind_power = 0
        turbulence_power = 0

    # Print Environment Parameters
    print(f"Stochastic: {args.stochastic}")
    print(f"Wind Power: {wind_power}")
    print(f"Turbulence Power: {turbulence_power}")
    print(f"Partially Observable: {args.partially_observable}")
    print(f"Render: {args.render}")
    print("\n\n")


    # Setup environment
    env: LunarLander = gym.make(
        "LunarLander-v2",
        continuous = True,
        gravity = -10.0,
        enable_wind = enable_wind,
        wind_power = wind_power,
        turbulence_power = turbulence_power,
        partially_observable = args.partially_observable,
        render_mode="human" if args.render else None,
        ref_lander = True,
        randomize_start = False,
    )
    env._max_episode_steps = 300
    env.reset(seed=seed)


    # Evaluate Controller
    scores = []
    for trial in tqdm(range(args.runs)):
        score, data = run(params, env, True)
        scores.append(score)
    env.close()
    data.graph()
    print(f"Average Score: {np.mean(scores)}" )

if __name__ == '__main__':
    main()
    