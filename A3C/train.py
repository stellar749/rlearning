import torch
import numpy as np
import gym
def train(cfg, envs, agent):

    print(f'Env:{cfg.env_name},Algorithm:{cfg.algo_name},Device:{cfg.device}')
    env = gym.make(cfg.env_name)
    env.seed(10)
    model = agent.model
    optimizer = agent.optimizer
    frame_idx = 0
    train_rewards = []
    train_ma_rewards = []
    steps = 0
    state = envs.reset()
    while frame_idx < cfg.max_frames:
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        #rollout trajectory
        for _ in range(cfg.n_steps):
            state = torch.FloatTensor(state).to(cfg.device)
            dist, value = model(state)
            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(cfg.device))
            masks.append(torch.FloatTensor(1-done).unsqueeze(1).to(cfg.device))
            state = next_state
            frame_idx += 1

            if frame_idx % 100 == 0:
                train_reward = np.mean([test_env(env, model, cfg) for _ in range(10)])
                print(f"frame_idx:{frame_idx}, train_reward:{train_reward}")
                train_rewards.append(train_reward)
                if train_ma_rewards:
                    train_ma_rewards.append(train_ma_rewards[-1]*0.9+train_reward*0.1)
                else:
                    train_ma_rewards.append(train_reward)
        next_state = torch.FloatTensor(next_state).to(cfg.device)
        _,next_value = model(next_state)
        returns = agent.compute_returns(next_value, rewards, masks)
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('complete training!')
    return train_rewards, train_ma_rewards   

def test_env(env, model, cfg, vis = False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(cfg.device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward