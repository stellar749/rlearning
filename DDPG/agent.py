from scipy import rand
import torch
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from DDPG.model import Actor, Critic, ReplayBuffer
import math
import random
import torch.nn as nn

class DDPG:
    def __init__(self, state_dim, action_dim, cfg):
        self.device = cfg.device
        self.critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = cfg.actor_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau
        self.gamma = cfg.gamma
    
    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0,0]

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # 转变为张量
        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action)
        expected_value = reward + (1 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        # 软更新
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
    
    def save(self, path):
        torch.save(self.actor.state_dict(), path+'ddpg_checkpoint.pt')
        print("successfully save!")
    
    def load(self, path):
        self.actor.load_state_dict(torch.load(path+'ddpg_checkpoint.pt'))
        print("successfully load!")