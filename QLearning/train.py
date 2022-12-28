def train(cfg, env, agent):
    print("begin trainning")
    print(f'env:{cfg.env_name}, algo:{cfg.algo_name}, device:{cfg.device}')
    rewards = []  
    ma_rewards = [] # moving average reward
    for i_ep in range(cfg.train_eps): # train_eps: 训练的最大episodes数
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
        agent.sample_count = 0
        while True:
            print(agent.sample_count)
            action = agent.choose_action(state)  # 根据算法选择一个动作
            next_state, reward, done, _ = env.step(action)  # 与环境进行一次动作交互
            agent.update(state, action, reward, next_state, done)  # Q-learning算法更新
            state = next_state  # 存储上一个观察值
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        print("Episode:{}/{}: reward:{:.1f}".format(i_ep+1, cfg.train_eps,ep_reward))
    print("complete trainning")
    return rewards, ma_rewards

def test(cfg, env, agent):
    print("begin tesing")
    print(f'env:{cfg.env_name}, algo:{cfg.algo_name}, device:{cfg.device}')
    i = 1
    for items in agent.Q_table:
        print("state:{}".format(i))
        i += 1
        for item in items:
            print(item)

    rewards = []  
    ma_rewards = [] # moving average reward
    for i_ep in range(cfg.test_eps): # train_eps: 训练的最大episodes数
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
        while True:
            action = agent.predict(state)  # 根据算法选择一个动作
            next_state, reward, done, _ = env.step(action)  # 与环境进行一次动作交互
            agent.update(state, action, reward, next_state, done)  # Q-learning算法更新
            state = next_state  # 存储上一个观察值
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        print("Episode:{}/{}: reward:{:.1f}".format(i_ep+1, cfg.test_eps,ep_reward))
    print("complete testing")
    return rewards, ma_rewards
