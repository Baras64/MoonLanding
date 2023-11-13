#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:18:26 2023 

@author: Aishik Pyne
"""
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm

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
    
class PIDOptimizationData():
    """tracks pid parameters and scores"""
    def __init__(self):
        self.params = []
        self.scores = []
        self.steps = []

    def add(self,params,score,step=None):
        self.params.append(params)
        self.scores.append(score)
        self.steps.append(step if step is not None else len(self.scores))

    def graph(self):
        """Plot Steps vs Score"""
        plt.plot(self.steps,self.scores)
        plt.grid()
        plt.title('PID Parameter Optimization')
        plt.ylabel('Score')
        plt.xlabel('Steps')
        plt.show()
    
    def save(self,filename):
        """Save data to file"""
        np.savez(filename, params=self.params, scores=self.scores, steps=self.steps)

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

def optimize(params, current_score, env, step):
    """ runs a step of randomized hill climbing """

    # add gaussian noise (less noise as n_steps increases)
    if step < 1000:
        test_params = params + np.random.normal(0,20.0,size=params.shape)
    else:
        test_params = params + np.random.normal(0,20.0/step,size=params.shape)

    # test params over 5 trial avg
    scores = []
    for trial in range(20):
        score,_ = run(test_params,env)
        scores.append(score)
    avg = np.mean(scores)
    
    # update params if improved
    if avg > current_score:
        return test_params,avg
    else:
        return params,current_score
    
def main():
    # Setup environment
    env: LunarLander = gym.make(
        "LunarLander-v2",
        continuous = True,
        gravity = -10.0,
        enable_wind = False,
        wind_power = 0,
        turbulence_power = 0,
        render_mode = None,
        randomize_start = False,
        ref_lander = True
    )
    env._max_episode_steps = 300
   
    # Seed RNGs
    np.random.seed(0)


    # Random Hill Climb over params
    params = np.array([0,0,0,0])
    score, _ = run(params, env, True)
    print(f"Initial Score: {score}")

    optimization_logger = PIDOptimizationData()
    for steps in range(1500):
        params,score = optimize(params,score,env,steps+1)
        optimization_logger.add(params,score)
        if steps%10 == 0:
            print(f"Step: {steps} Score: {score}, Params: {params}")
    optimization_logger.graph()

    # Save optimization data
    optimization_logger.save('pid_optimization.npz')

if __name__ == '__main__':
    main()
    