from sqlalchemy import false
from DDPG.env import OUNoise


from DDPG.env import NormalizedActions, OUNoise
def train(cfg, env, agent):
    print(f'Env:{cfg.env_name},Algorithm:{cfg.algo_name},Device:{cfg.device}')
    ou_noise = OUNoise(env.action_space)
    rewards = []
    ma_rewards = []

    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ou_noise.reset()
        done = False
        i_step = 0
        ep_reward = 0
        while not done:
            i_step += 1
            action = agent.choose_action(state)
            action = ou_noise.get_action(action, i_step)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            agent.update()
            ep_reward += reward
        if (i_ep + 1) % cfg.target_update == 0: #更新target
            agent.target_critic.load_state_dict(agent.critic.state_dict())
            agent.target_actor.load_state_dict(agent.actor.state_dict())
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
        done = False
        ep_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"epoch:{i_ep + 1} / {cfg.test_eps}, reward:{ep_reward:.2f}")
    print('complete eval!')
    return rewards, ma_rewards    