
def train(cfg, env, agent):
    print(f'Env:{cfg.env_name},Algorithm:{cfg.algo_name},Device:{cfg.device}')

    rewards = []
    ma_rewards = []

    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            agent.update()
            ep_reward += reward
            if done:
                break;
        if (i_ep + 1) % cfg.target_update == 0: #更新target
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print(f"epoch:{i_ep + 1} / {cfg.train_eps}, reward:{ep_reward:.2f}")
    print('complete training!')
    env.close()
    return rewards, ma_rewards

def eval(cfg, env, agent):
    print(f'Env:{cfg.env_name},Algorithm:{cfg.algo_name},Device:{cfg.device}')
    rewards = []
    ma_rewards = []

    for i_ep in range(cfg.test_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if done:
                break;
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"epoch:{i_ep + 1} / {cfg.test_eps}, reward:{ep_reward:.2f}")
    print('complete eval!')
    return rewards, ma_rewards    