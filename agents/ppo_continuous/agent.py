

from agents import BaseAgent
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from .cont_model import Actor_Model

class PPOAgentCont(BaseAgent):
    def __init__(self, env_name, env):
        self.device = None
        self.env = env
        self.action_size = self.env.action_space.shape[0]
        self.state_size = self.env.observation_space.shape
        self.lr = 0.00025
        self.optimizer = Adam
        
        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)
        
        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        
        self.Actor.Actor.load_weights(f"/home/harsh/Downloads/AIP/MoonLanding/agents/ppo_continuous/{env_name}_PPO_Actor.h5")

    def action(self, state):
        state = np.reshape(state, [1, self.state_size[0]])
        
        pred = self.Actor.predict(state)

        low, high = -1.0, 1.0 # -1 and 1 are boundaries of tanh
        action = pred + np.random.uniform(low, high, size=pred.shape) * self.std
        action = np.clip(action, low, high)
        
        return action[0]        