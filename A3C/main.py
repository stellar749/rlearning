import sys, os
import gym
import torch
import datetime

curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间

from A3C.agent import A3C
from A3C.utils import plot_rewards
from A3C.utils import save_results,make_dir
from A3C.train import train
from A3C.multiprocessing_env import SubprocVecEnv

algo_name = "A3C"  # 算法名称
env_name = "CartPole-v0"  # 环境名称
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测GPU

class A3CConfig:
    def __init__(self):
        self.algo_name = algo_name # 算法名称
        self.env_name = env_name # 环境名称
        self.n_envs = 8 #异步的环境数目
        self.gamma = 0.99 # reward的衰减率
        self.hidden_dim = 256
        self.lr = 1e-3
        self.max_frames = 30000
        self.n_steps = 5

        self.train_eps = 200 # 训练的回合数
        self.test_eps = 20 # 测试的回合数
        self.batch_size = 5

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
'''
def env_agent_config(cfg, seed = 1):
    env = gym.make(cfg.env_name)
    env.seed = seed
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, action_dim, cfg)
    return env, agent
'''

def make_envs(env_name):
    def _thunk():
        env = gym.make(env_name)
        env.seed(2)
        return env
    return _thunk

if __name__ == "__main__":
    cfg = A3CConfig()
    plot_cfg = PlotConfig()
    #train
    envs = [make_envs(cfg.env_name) for i in range(cfg.n_envs)]
    envs = SubprocVecEnv(envs)
    state_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.n
    agent = A3C(state_dim, action_dim, cfg)

    rewards, ma_rewards = train(cfg, envs, agent)
    make_dir(cfg.result_path, cfg.model_path)
    #agent.save(path = cfg.model_path)
    save_results(rewards, ma_rewards, tag='train',
            path=cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")  # 画出结果

'''
    #eval
    env, agent = env_agent_config(cfg, seed = 10)
    agent.load(path = cfg.model_path)
    rewards, ma_rewards = eval(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='eval',
            path=cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="eval")  # 画出结果
'''
