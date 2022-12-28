import torch.nn as nn
import torch.nn.functional as F
import random
import torch

class Actor(nn.Module):
    def __init__(self, state_dim,action_dim,hidden_dim=128, init_w = 3e-3):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim) 
        self.linear2 = nn.Linear(hidden_dim,hidden_dim) 
        self.linear3 = nn.Linear(hidden_dim, action_dim)

        self.linear3.weight.data.uniform_ (-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.linear1(x)) 
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim,action_dim,hidden_dim=128, init_w = 3e-3):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim) 
        self.linear2 = nn.Linear(hidden_dim,hidden_dim) 
        self.linear3 = nn.Linear(hidden_dim, 1) 
        
        self.linear3.weight.data.uniform_ (-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) #按维数1拼接
        x = F.relu(self.linear1(x)) 
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position+1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
