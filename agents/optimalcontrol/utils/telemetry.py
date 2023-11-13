import numpy as np
from matplotlib import pyplot as plt
from optimalcontrol.state import Trajectory, State, Action


class Telemetry:

    def __init__(self) -> None:
        self.fig = plt.figure(1, figsize=(20, 4))

        gs = self.fig.add_gridspec(2,5)

        self.ax_lander_x = self.fig.add_subplot(gs[0, 0])
        self.ax_lander_y = self.fig.add_subplot(gs[0, 1])
        self.ax_lander_theta = self.fig.add_subplot(gs[0, 2])
        self.ax_vel_x = self.fig.add_subplot(gs[1, 0])
        self.ax_vel_y = self.fig.add_subplot(gs[1, 1])
        self.ax_vel_theta = self.fig.add_subplot(gs[1, 2])

        self.ax_map = self.fig.add_subplot(gs[:, 3:])

        self.fig.tight_layout()


    def clear(self):
        # Clear all axis
        self.ax_lander_x.clear()
        self.ax_lander_y.clear()
        self.ax_lander_theta.clear()
        self.ax_vel_x.clear()
        self.ax_vel_y.clear()
        self.ax_vel_theta.clear()

    def update(self, trajectory: Trajectory):

        self.clear()

        self.ax_lander_x.plot(trajectory.t, trajectory.lander_x, label="lander_x")
        self.ax_lander_x.set_title("lander_x")
        self.ax_lander_x.set_xlim(0, max(trajectory.size, 50))
        self.ax_lander_x.set_ylim(0, 2)

        self.ax_lander_y.plot(trajectory.t, trajectory.lander_y, label="lander_y")
        self.ax_lander_y.set_title("lander_y")
        self.ax_lander_y.set_xlim(0, max(trajectory.size, 50))
        self.ax_lander_y.set_ylim(0, 2)

        self.ax_lander_theta.plot(trajectory.t, trajectory.lander_theta, label="lander_theta")
        self.ax_lander_theta.set_title("lander_theta")
        self.ax_lander_theta.set_ylim(-np.pi, np.pi)
        self.ax_lander_theta.set_xlim(0, max(trajectory.size, 50))

        self.ax_vel_x.plot(trajectory.t, trajectory.vel_x, label="vel_x")
        self.ax_vel_x.set_title("vel_x")
        self.ax_vel_x.set_ylim(-2, 2)
        self.ax_vel_x.set_xlim(0, max(trajectory.size, 50))

        self.ax_vel_y.plot(trajectory.t, trajectory.vel_y, label="vel_y")
        self.ax_vel_y.set_title("vel_y")
        self.ax_vel_y.set_ylim(-2, 2)
        self.ax_vel_y.set_xlim(0, max(trajectory.size, 50))

        self.ax_vel_theta.plot(trajectory.t, trajectory.vel_theta, label="vel_theta")
        self.ax_vel_theta.set_title("vel_theta")
        self.ax_vel_theta.set_ylim(-np.pi, np.pi)
        self.ax_vel_theta.set_xlim(0, max(trajectory.size, 50))
        
        # Figure Tight Layout
        
        plt.pause(0.001)
