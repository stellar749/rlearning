from itertools import count

from sympy import epath
def train(cfg, env, agent):
    print(f'Env:{cfg.env_name},Algorithm:{cfg.algo_name},Device:{cfg.device}')
    state_pool = []
    action_pool = []
    reward_pool = []
    rewards = []
    ma_rewards = []

    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        for _ in count():
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                reward = 0
            state_pool.append(state)
            action_pool.append(action)
            reward_pool.append(reward)
            state = next_state
            if done:
                print('Episode:',i_ep,'Reward:',ep_reward)
                break
        if i_ep > 0 and i_ep % cfg.batch_size == 0: #每个batch进行更新
            agent.update(reward_pool, state_pool, action_pool)
            state_pool = []
            action_pool = []
            reward_pool = []
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
    print('complete training!')
    return rewards, ma_rewards

def eval(cfg, env, agent):
    print(f'Env:{cfg.env_name},Algorithm:{cfg.algo_name},Device:{cfg.device}')
    rewards = []
    ma_rewards = []

    for i_ep in range(cfg.test_eps):
        state = env.reset()
        ep_reward = 0
        for _ in count():
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                reward = 0
            state = next_state
            if done:
                print('Episode:',i_ep,'Reward:',ep_reward)
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
    print('complete eval!')
    return rewards, ma_rewards    