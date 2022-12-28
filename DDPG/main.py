import sys, os
import gym
import torch
import datetime

curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间

from DDPG.agent import DDPG
from DDPG.utils import plot_rewards
from DDPG.utils import save_results,make_dir
from DDPG.train import train,eval
from DDPG.env import NormalizedActions, OUNoise

algo_name = "ddpg"  # 算法名称
env_name = "Pendulum-v1"  # 环境名称
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测GPU

class Config:
    def __init__(self):
        self.algo_name = algo_name # 算法名称
        self.env_name = env_name # 环境名称
        self.device = device
        self.seed = 10
        self.train_eps = 300 # 训练的回合数
        self.test_eps = 20 # 测试的回合数
        
        self.gamma = 0.99 # reward的衰减率
        self.critic_lr = 1e-4 # 学习率
        self.actor_lr = 1e-4 # 学习率
        self.memory_capacity = 8000
        self.batch_size = 128
        self.target_update = 2
        self.hidden_dim = 256
        self.soft_tau = 1e-2

        self.model_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/models/'  # 保存模型的路径
        self.result_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/results/'  # 保存结果的路径

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
    env = NormalizedActions(gym.make(cfg.env_name)) # 装饰action噪声
    env.seed(seed) # 随机种子
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = DDPG(n_states,n_actions,cfg)
    return env, agent

if __name__ == "__main__":
    cfg = Config()
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


