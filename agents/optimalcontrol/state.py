import math
import numpy as np
from typing import List
from matplotlib import pyplot as plt

class State:

    def __init__(self, observation):
        lander_x, lander_y, vel_x, vel_y, lander_theta, vel_theta, left_leg, right_leg, new_x1, new_y1, new_x2, new_y2 = observation

        self.lander_x = lander_x
        self.lander_y = lander_y
        self.lander_theta = lander_theta
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.vel_theta = vel_theta

        self.left_leg = left_leg
        self.right_leg = right_leg

        self.goal_post_left_x = new_x1
        self.goal_post_left_y = new_y1
        self.goal_post_right_x = new_x2
        self.goal_post_right_y = new_y2
        
    @property
    def vel(self):
        return math.hypot(self.vel_x, self.vel_y)
    
    @property
    def vector(self):
        return np.array([self.lander_x, self.lander_y, self.lander_theta, self.vel_x, self.vel_y, self.vel_theta])

class Action:
    
    def __init__(self, action):
        self.left_thruster = action[0]
        self.right_thruster = action[1]
    
    @property
    def vector(self):
        return np.array([self.left_thruster, self.right_thruster])

class Trajectory:
    """Sequence of (states, action, next_states) tuples
    """

    def __init__(self):

        self.states: List[State] = []
        self.actions = []
        self.next_states = []
        self.t = []

        # self.lander_x = []
        # self.lander_y = []
        # self.lander_theta = []
        # self.vel_x = []
        # self.vel_y = []
        # self.vel_theta = []
        # self.t = []

    @property
    def size(self):
        return len(self.t)

    @property
    def lander_x(self):
        return [s.lander_x for s in self.states]
    
    @property
    def lander_y(self):
        return [s.lander_y for s in self.states]
    
    @property
    def lander_theta(self):
        return [s.lander_theta for s in self.states]
    
    @property
    def vel_x(self):
        return [s.vel_x for s in self.states]
    
    @property
    def vel_y(self):
        return [s.vel_y for s in self.states]
    
    @property
    def vel_theta(self):
        return [s.vel_theta for s in self.states]

    def recent_history(self, window_size=10):
        """Returns the most recent window_size states
        """
        return self.states[-window_size:], self.actions[-window_size:], self.next_states[-window_size:]
        
    
    def append(self, state: State, action: Action, next_state: State, t: int = None):
        self.t.append(t) if t is not None else self.t.append(self.size)

        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)


       

def render_trajectory(trajectory: Trajectory):

    # Matplot lib create new plot
    if plt.fignum_exists(100):
        fig = plt.figure(100)
        axs = fig.get_axes()
        # Clear all axes
        for ax in axs:
            ax.clear()
    else:
        # Figure is closed
        fig = plt.figure(100)
        axs = fig.subplots(2, 3).flatten()

    axs[0].plot(trajectory.t, trajectory.lander_x, label="lander_x")
    axs[0].set_title("lander_x")
    axs[0].set_xlim(0, max(trajectory.size, 50))

    axs[1].plot(trajectory.t, trajectory.lander_y, label="lander_y")
    axs[1].set_title("lander_y")
    axs[1].set_xlim(0, max(trajectory.size, 50))

    axs[2].plot(trajectory.t, trajectory.lander_theta, label="lander_theta")
    axs[2].set_title("lander_theta")
    axs[2].set_ylim(-np.pi, np.pi)
    axs[2].set_xlim(0, max(trajectory.size, 50))

    axs[3].plot(trajectory.t, trajectory.vel_x, label="vel_x")
    axs[3].set_title("vel_x")
    axs[3].set_ylim(-2, 2)
    axs[3].set_xlim(0, max(trajectory.size, 50))

    axs[4].plot(trajectory.t, trajectory.vel_y, label="vel_y")
    axs[4].set_title("vel_y")
    axs[4].set_ylim(-2, 2)
    axs[4].set_xlim(0, max(trajectory.size, 50))

    axs[5].plot(trajectory.t, trajectory.vel_theta, label="vel_theta")
    axs[5].set_title("vel_theta")
    axs[5].set_ylim(-np.pi, np.pi)
    axs[5].set_xlim(0, max(trajectory.size, 50))
    
    # Figure Tight Layout
    fig.tight_layout()
    plt.pause(0.001)