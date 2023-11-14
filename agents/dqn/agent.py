from agents import BaseAgent
import torch

class DQNAgent(BaseAgent):
    def __init__(self, env_name):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if env_name == "discreteaction-deterministic-fo":
            self.policy_net = torch.load("/home/harsh/Downloads/AIP/MoonLanding/agents/dqn/discreteaction-deterministic-fo_policy_net.pt").to(self.device)
        else:
            self.policy_net = torch.load("/home/harsh/Downloads/AIP/MoonLanding/agents/dqn/discreteaction-stochastic-fo_policy_net.pt").to(self.device)
    
    def action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.policy_net(state).max(1)[1].view(1, 1)
        return action.item()