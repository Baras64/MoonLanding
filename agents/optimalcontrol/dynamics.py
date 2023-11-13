import gymnasium as gym
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from gymnasium.envs.box2d.lunar_lander import LunarLander
from typing import List
from optimalcontrol.state import State, Action, Trajectory

def fit_linear_regression(X, y):
    """
    Fits a linear regression model to the input data X and target variable y.
    
    Args:
        X (array-like): Input data of shape (n_samples, n_features)
        y (array-like): Target variable of shape (n_samples,)
    
    Returns:
    model (LinearRegression): Fitted linear regression model
    """
    model = LinearRegression()
    model.fit(X, y)
    # Print fitting error and summary
    print(f"Training Fit error: {model.score(X, y)}")
    return model

def fit_mlp(X, y):
    """Fits a multilayer perceptron to the input data X and target variable y.

    Args:
        X (array-like): Input data of shape (n_samples, n_features)
        y (array-like): Target variable of shape (n_samples,)

    Returns:
        MLPRegressor: Fitted mlp model
    """
    # Create a neural network with 2 hidden layers of 10 neurons each
    model = MLPRegressor(hidden_layer_sizes=(100, 10), max_iter=10000)

    # Fit the neural network to the data
    model.fit(X, y)

    # Print fitting error and summary
    print(f"Training Fit error: {model.score(X, y)}")
    return model 

def print_state(observation):
    lander_x, lander_y, vel_x, vel_y, lander_theta, vel_theta, left_leg, right_leg, new_x1, new_y1, new_x2, new_y2 = observation 
    print(f"Pos \t{lander_x:4f} {lander_y:4f} {lander_theta:4f}")
    print(f"Vel \t{vel_x:4f} {vel_y:4f} {vel_theta:4f}")
    print(f"Contact \t{left_leg:0f} {right_leg:0f}")


class Transition:

    def __init__(self, x, u, x_next):
        self.hist_window = 10
        self.look_ahead = 10

        self.model = LinearRegression()

    def fit(self, state_history: List[State],
            action_history: List[Action],
            next_state_history: List[State]):
        
        state_history = np.array([s.vector for s in state_history])
        action_history = np.array([a.vector for a in action_history])
        next_state_history = np.array([s.vector for s in next_state_history])

        X = np.concatenate([state_history, action_history], axis=1)
        y = next_state_history

        self.model.fit(X, y)

        # Print fitting error and summary
        print(f"Training Fit error: {self.model.score(X, y)}")
        
    def predict(self, state: State, action: Action):
        # Np Stack two 1D vectors
        next_state = self.model.predict(np.concatenate([state.vector, action.vector]))

        return next_state
    
if __name__ == "__main__":

    env: LunarLander = gym.make(
        "LunarLander-v2",
        continuous = True,
        gravity = -10.0,
        enable_wind = False,
        wind_power = 0,
        turbulence_power = 0,
        # render_mode="human", # Turn off for training
        randomize_start = True,
        ref_lander = True
    )
    print(f"Action Space: {env.action_space}")
    env.action_space.seed(42)

    """
    lander_x,
    lander_y,
    vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
    vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
    self.lander.angle,
    20.0 * self.lander.angularVelocity / FPS,
    1.0 if self.legs[0].ground_contact else 0.0,
    1.0 if self.legs[1].ground_contact else 0.0,
    new_x1,
    new_y1,
    new_x2,
    new_y2
    """

    dimX, dimY = 8, 6
    trainX, trainY = [], []

    trajectory = []

    observation, info = env.reset(seed=42)
    for t in range(10):    
        action = env.action_space.sample() # [0, 0]
        action = [0, 0]

        

        next_observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            next_observation, info = env.reset()
        else:
            lander_x, lander_y, vel_x, vel_y, lander_theta, vel_theta, left_leg, right_leg, new_x1, new_y1, new_x2, new_y2 = observation 
            xu = np.concatenate([observation[:8], np.array(action)], axis=0)
            trainX.append(xu)
            trainY.append(next_observation[:6])

        observation = next_observation

    # Convert to numpy arrays
    trainX = np.array(trainX)
    trainY = np.array(trainY)

    # Print shape of buffers
    print(f"Buffer X {trainX.shape} Y {trainY.shape}")



    # Train transition function
    transition_fns = {
        "lr": fit_linear_regression(trainX, trainY),
        "mlp": fit_mlp(trainX, trainY)
    }




    testX, testY = [], []

    # Test transition function
    observation, info = env.reset(seed=24)
    for t in range(10000):    
        action = env.action_space.sample()
        action = [0, 0]

        next_observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            next_observation, info = env.reset()
        else:
            xu = np.concatenate([observation[:8], np.array(action)], axis=0)
            testX.append(xu)
            testY.append(next_observation[:6])
        observation = next_observation

    # Convert to numpy arrays
    testX = np.array(testX)
    testY = np.array(testY)

    # Print Train Error
    print(f"Train error: {np.sqrt(np.square(transition_fns['lr'].predict(trainX) - trainY).mean(axis=0))}")

    # Print Prediction Error
    print(f"Test error: {np.sqrt(np.square(transition_fns['lr'].predict(testX) - testY).mean(axis=0))}")

    # Print Train Error
    print(f"Train error: {np.sqrt(np.square(transition_fns['mlp'].predict(trainX) - trainY).mean(axis=0))}")

    # Print Prediction Error
    print(f"Test error: {np.sqrt(np.square(transition_fns['mlp'].predict(testX) - testY).mean(axis=0))}")

    env.close()
