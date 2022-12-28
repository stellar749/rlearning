import imp
from multiprocessing.dummy import active_children
import os
import torch
import torch.optim as optim
import numpy as np
from PPO.model import Actor, Critic
from PPO.memory import PPOMemory 

class PPO:
    def __init__(self, state_dim, action_dim, cfg):
        self.gamma = cfg.gamma
        self.continuous = cfg.continuous
        self.policy_clip = cfg.policy_clip
        self.n_epochs = cfg.n_epochs #
        self.gae_lambda = cfg.gae_lambda
        self.device = cfg.device
        self.actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.critic = Critic(state_dim, cfg.hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = cfg.critic_lr)
        self.memory = PPOMemory(cfg.batch_size)
        self.loss = 0
    
    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        dist = self.actor(state) #生成该state下的Categorical对象
        value = self.critic(state) # 没有体现action ?
        action = dist.sample() #按照action概率抽样
        probs = torch.squeeze(dist.log_prob(action)).item() # 取action的概率的对数
        if self.continuous:
            action = torch.tanh(action)
        else:
            action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return action, probs, value

    def update(self):
        for _ in range(self.n_epochs): #利用旧参数下采样的数据，对新参数进行n次改进
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.sample()
            values = vals_arr[:]
            # compute advantage
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1-int(dones_arr[k])) - values[k])
                    discount = self.gae_lambda * discount
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)
            #SGD
            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch], dtype=torch.float).to(self.device)
                actions = torch.tensor(action_arr[batch],dtype=torch.float).to(self.device)
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighed_probs = advantage[batch] * prob_ratio
                weighed_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+ self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighed_probs, weighed_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2 # values(returns)不会再变了，但是critic_value会不断更新,这里是用MC的方法来对Q-function进行估计
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5 * critic_loss
                self.loss = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()


    def save(self, path):
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint = os.path.join(path, 'ppo_critic.pt')
        torch.save(self.actor.state_dict(), actor_checkpoint)
        torch.save(self.critic.state_dict(), critic_checkpoint)
        print("successfully save!")
    
    def load(self, path):
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint = os.path.join(path, 'ppo_critic.pt')
        self.actor.load_state_dict(torch.load(actor_checkpoint))
        self.critic.load_state_dict(torch.load(critic_checkpoint))
        print("successfully load!")