import numpy as np 
import math
import torch
from collections import defaultdict

class QLearning:
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.lr = cfg.lr
        self.gamma = cfg.gamma
        self.sample_count = 0
        self.epsilon = 0
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table = np.zeros((state_dim, action_dim))
    
    def choose_action(self, state):
        #epsilon decay
        self.sample_count += 1
        self.epsilon = (self.epsilon_end - self.epsilon_start) * math.exp(-1 * self.sample_count / self.epsilon_decay)
        if np.random.uniform(0,1) > self.epsilon:
            action = np.argmax(self.Q_table[state-1][:])
        else:
            action = np.random.choice(self.action_dim)
        return action

    def predict(self, state):
        action = np.argmax(self.Q_table[state-1][:]) 
        return action

    def update(self, state, action, reward, next_state, done):
        Q_predict = self.Q_table[state-1][action]
        if done:
            Q_target = reward
        else:
            Q_target = reward + np.max(self.Q_table[next_state-1][:]) * self.gamma
        self.Q_table[state-1][action] += self.lr * (Q_target - Q_predict)
    
    def save(self, path):
        import dill
        torch.save(
            obj = self.Q_table,
            f = path + "Qlearning_model.pkl",
            pickle_module = dill
        )
        print("successfully save!")
    
    def load(self, path):
        import dill
        self.Q_table = torch.load(f = path + "Qlearning_model.pkl", pickle_module= dill)
        print("successfully load!")
