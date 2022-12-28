import sys, os
import gym
import torch
import datetime

curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间

from DQN.agent import DQN
from DQN.utils import plot_rewards
from DQN.utils import save_results,make_dir
from DQN.train import train,eval

algo_name = "DQN"  # 算法名称
env_name = "CartPole-v0"  # 环境名称
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测GPU

class DQNConfig:
    def __init__(self):
        self.algo_name = algo_name # 算法名称
        self.env_name = env_name # 环境名称
        self.train_eps = 200 # 训练的回合数
        self.test_eps = 30 # 测试的回合数
        
        self.gamma = 0.95 # reward的衰减率
        self.epsilon_start = 0.90
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        self.lr = 0.0001 # 学习率
        self.memory_capacity = 500
        self.batch_size = 64
        self.target_update = 4
        self.hidden_dim = 256

        self.model_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/models/'  # 保存模型的路径
        self.result_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/results/'  # 保存结果的路径
        self.device = device

class PlotConfig:
    ''' 绘图相关参数设置
    '''
    def __init__(self) -> None:
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = device # 检测GPU
        self.result_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/results/'  # 保存结果的路径
        self.save = True  # 是否保存图片

def env_agent_config(cfg, seed = 1):
    env = gym.make(cfg.env_name)
    env.seed = seed
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, action_dim, cfg)
    return env, agent

if __name__ == "__main__":
    cfg = DQNConfig()
    plot_cfg = PlotConfig()
    #train
    env, agent = env_agent_config(cfg, seed = 1)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path = cfg.model_path)
    save_results(rewards, ma_rewards, tag='train',
            path=cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")  # 画出结果

    #eval
    env, agent = env_agent_config(cfg, seed = 10)
    agent.load(path = cfg.model_path)
    rewards, ma_rewards = eval(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='eval',
            path=cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="eval")  # 画出结果


