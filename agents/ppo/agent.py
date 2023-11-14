

from agents import BaseAgent
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from .model import Actor_Model

class PPOAgentEval(BaseAgent):
    def __init__(self, env_name, env):
        self.device = None
        self.env = env
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.lr = 0.00025
        self.optimizer = Adam
        
        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        
        if env_name == "discreteaction-deterministic-fo":
            self.Actor.Actor.load_weights(f"/home/harsh/Downloads/AIP/MoonLanding/agents/ppo/{env_name}_PPO_Actor.h5")
        else:
            self.Actor.Actor.load_weights(f"/home/harsh/Downloads/AIP/MoonLanding/agents/ppo/{env_name}_PPO_Actor.h5")
    
    def action(self, state):
        state = np.reshape(state, [1, self.state_size[0]])
        return np.argmax(self.Actor.predict(state)[0])
        