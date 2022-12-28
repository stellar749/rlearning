from scipy import rand
import torch
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from DQN.model import MLP, ReplayBuffer
import math
import random

class DQN:
    def __init__(self, state_dim, action_dim, cfg):
        self.gamma = cfg.gamma
        self.action_dim = action_dim
        self.device = cfg.device

        self.frame_idx = 0
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-1. * frame_idx / cfg.epsilon_decay)

        self.batch_size = cfg.batch_size
        self.policy_net = MLP(state_dim, action_dim,hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim,hidden_dim=cfg.hidden_dim).to(self.device)
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = cfg.lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
    
    def choose_action(self, state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)
        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)  
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)  
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)

        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()

        expected_q_values = reward_batch + self.gamma * next_q_values * (1-done_batch)
        loss = torch.nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1) #防止梯度爆炸
        self.optimizer.step()
    
    def save(self, path):
        torch.save(self.target_net.state_dict(), path+'dqn_checkpoint.pt')
        print("successfully save!")
    
    def load(self, path):
        self.target_net.load_state_dict(torch.load(path+'dqn_checkpoint.pt'))
        print("successfully load!")
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)