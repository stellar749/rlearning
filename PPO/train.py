def train(cfg, env, agent):
    print(f'Env:{cfg.env_name},Algorithm:{cfg.algo_name},Device:{cfg.device}')
    rewards = []
    ma_rewards = []
    steps = 0

    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, prob, val = agent.choose_action(state)
            state_, reward, done,_ = env.step(action)
            steps += 1
            ep_reward += reward
            agent.memory.push(state, action, prob, val, reward, done)
            if steps % cfg.update_fre == 0:
                agent.update()
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print(f"epoch:{i_ep + 1} / {cfg.train_eps}, reward:{ep_reward:.2f}")
    print('complete training!')
    return rewards, ma_rewards

def eval(cfg, env, agent):
    print(f'Env:{cfg.env_name},Algorithm:{cfg.algo_name},Device:{cfg.device}')
    rewards = []
    ma_rewards = []

    for i_ep in range(cfg.test_eps):
        state = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action, prob, val = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            ep_reward += reward
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"epoch:{i_ep + 1} / {cfg.test_eps}, reward:{ep_reward:.2f}")
    print('complete eval!')
    return rewards, ma_rewards    