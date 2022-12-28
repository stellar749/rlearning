import torch
from torch.distributions import Bernoulli
from torch.autograd import Variable
import numpy as np
from PolicyGradient.model import MLP

class PolicyGradient:
    def __init__(self, state_dim, cfg):
        self.gamma = cfg.gamma
        self.policy_net = MLP(state_dim, hidden_dim = cfg.hidden_dim)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr = cfg.lr)
    
    def choose_action(self, state):
        state = torch.from_numpy(state).float()
        state = Variable(state)
        probs = self.policy_net(state)
        m = Bernoulli(probs)
        action = m.sample()
        action = action.data.numpy().astype(int)[0] #转为标量
        return action

    def update(self, reward_pool, state_pool, action_pool):
        #discounted award
        running_add = 0
        # assign suitable credit
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == 0: #对reward_add进行清零
                running_add = 0 
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add
        # normalize reward, add baseline
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std
        
        #gradient descent
        self.optimizer.zero_grad()
        for i in range(len(reward_pool)):
            state = state_pool[i]
            state = Variable(torch.from_numpy(state).float())
            action = Variable(torch.FloatTensor([action_pool[i]]))
            reward = reward_pool[i]
            probs = self.policy_net(state)
            m = Bernoulli(probs)
            loss = -m.log_prob(action) * reward #negative loss function
            loss.backward()
        self.optimizer.step()
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path+'pg_checkpoint.pt')
        print("successfully save!")
    
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path+'pg_checkpoint.pt'))
        print("successfully load!")