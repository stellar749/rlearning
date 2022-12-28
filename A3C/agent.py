import os
import torch
import torch.optim as optim
import numpy as np
from A3C.model import ActorCritic

class A3C:
    def __init__(self, state_dim, action_dim, cfg):
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.model = ActorCritic(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
    
    def compute_returns(self, next_values, rewards, masks):
        R = next_values
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0,R)
        return returns


    def save(self, path):
        model_checkpoint = os.path.join(path, 'a3c.pt')
        torch.save(self.actor.state_dict(), model_checkpoint)
        print("successfully save!")
    
    def load(self, path):
        model_checkpoint = os.path.join(path, 'a3c.pt')
        self.actor.load_state_dict(torch.load(model_checkpoint))
        print("successfully load!")