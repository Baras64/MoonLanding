import numpy as np
from matplotlib import pyplot as plt

class GridSystem:
    def __init__(self, wind=None):
        self.x_max = 10
        self.y_max = 10
        self.x_min = -10
        self.y_min = -10

        self.max_control = 10
        self.max_wind = 2


        self.wind = wind
        
        
        self.history_window = 10

        self.fig, self.axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        self.axs = self.axs.flatten()

        self.reset()

    def reset(self):
        # self.reference = np.random.uniform(-10, 10, size=(2,))
        self.reference = np.array([5, 5])
        self.x = -5
        self.y = -5
        self.state_history = np.repeat(self.state.reshape(1, -1), self.history_window, axis=0)
        self.error_history = np.zeros((self.history_window, 2))

        if self.wind == 'horizontal':
            self.wind_mag = self.max_wind
            self.wind_dir = np.random.choice([0, np.pi])
        elif self.wind == 'vertical':
            self.wind_mag = self.max_wind
            self.wind_dir = np.random.choice([-np.pi/2, np.pi/2])
        elif self.wind == 'random':
            self.wind_mag = np.random.uniform(0, self.max_wind)
            self.wind_dir = np.random.uniform(-np.pi, np.pi)
        else:
            self.wind_mag = 0
            self.wind_dir = 0 
        self.wind_fn = lambda x: np.array([self.wind_mag * np.cos(self.wind_dir), self.wind_mag * np.sin(self.wind_dir)])

    def reset_reference(self):
        self.reference = np.random.uniform(-10, 10, size=(2,))
        
    @property
    def state(self):
        return np.array([self.x, self.y])
    
    def update(self, control):
        
        self.x += np.clip(control, -self.max_control, self.max_control)[0] 
        self.y += np.clip(control, -self.max_control, self.max_control)[1]
        # self.x += control[0] 
        # self.y += control[1]

        if self.wind:
            self.x += self.wind_fn(self.state)[0]
            self.y += self.wind_fn(self.state)[1]

        self.x = max(self.x_min, min(self.x, self.x_max))
        self.y = max(self.y_min, min(self.y, self.y_max))

        self.error_history = np.vstack([self.error_history, self.reference - self.state])
        self.state_history = np.vstack([self.state_history[1:], self.state])
       
        return self.state
    
    
        
    def render(self):


        # Clear the current figure
        # plt.cla()
        self.axs[0].clear()
        
        # for stopping simulation with the esc key.
        self.fig.canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        self.fig.canvas.mpl_connect(
            'key_release_event',
            lambda event: [self.reset() if event.key == 'r' else None])
        
        # Draw marker for goal
        self.axs[0].plot(self.state[0], self.state[1], "or", label="agent")
        self.axs[0].plot(self.reference[0], self.reference[1], "xg", label="reference")

        self.axs[0].plot(self.state_history[:, 0], self.state_history[:, 1], "--b", label="trajectory")

        self.axs[0].grid(True)
        self.axs[0].set_aspect('equal', adjustable='box')

        self.axs[0].set_ylim(self.y_min, self.y_max)
        self.axs[0].set_xlim(self.x_min, self.x_max)
        # Place legend outside of plot
        self.axs[0].legend(loc='upper left', borderaxespad=0.)
        self.axs[0].set_title(f"Error {np.around(np.linalg.norm(self.error_history[-1]),3)}")


        self.axs[1].clear()

        # Plot Wind
        circle = plt.Circle((0, 0), 1, color='g', fill=False)
        self.axs[1].add_artist(circle)
        self.axs[1].arrow(0, 0, \
                        self.wind_mag*np.cos(self.wind_dir)/self.max_wind, \
                        self.wind_mag*np.sin(self.wind_dir)/self.max_wind, \
                        width=0.03, color='r', length_includes_head=True)

        self.axs[1].grid(True)
        self.axs[1].set_aspect('equal', adjustable='box')
        self.axs[1].set_ylim(-1.5, 1.5)
        self.axs[1].set_xlim(-1.5, 1.5)
        self.axs[1].set_title(f"Wind ")

        plt.pause(0.001)
