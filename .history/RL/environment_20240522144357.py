import gym
from gym import spaces
import numpy as np
import deque

from src.simulator.Simulator_platform import Simulator_Platform
from src.simulator.config import ConfigManager

class ManhattanTrafficEnv(gym.Env):
    """定义曼哈顿路网模拟环境"""
    
    def __init__(self):
        super(ManhattanTrafficEnv, self).__init__()
        # 定义动作空间为一个连续值，代表调整的系数，比如从-0.1到0.1的增减
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32)
        
        # 定义状态空间，假设包括车辆分布的几个关键数值，以及请求接受和拒绝的数量
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
        
        # initial status
        self.state = None
        self.config = ConfigManager()
        self.simulator = Simulator_Platform(0, self.config)  # Sim start time, ConfigManager

        self.n_steps_delay = 40 # 40 steps delay. one step is 15 seconds, so 10 minutes delay
        self.past_actions = deque(maxlen=self.n_steps_delay)
        self.past_rejections = deque(maxlen=self.n_steps_delay)

        self.decay_factor = self.config.get('DECAY_FACTOR')

        # initialize the config
        self.init_config({'REWARD_THETA': 1.0, 'REWARD_TYPE': 'REJ', 'NODE_LAYERS': 2, 'MOVING_AVG_WINDOW': 20, 'DECAY_FACTOR': 0.9})

    def init_config(self, args: dict):
        self.change_config(self.config, args) # change the config based on the args 

        REWARD_THETA = self.config.get('REWARD_THETA')
        REWARD_TYPE = self.config.get('REWARD_TYPE')
        NODE_LAYERS = self.config.get('NODE_LAYERS')
        MOVING_AVG_WINDOW = self.config.get('MOVING_AVG_WINDOW')
        DECAY_FACTOR = self.config.get('DECAY_FACTOR')

        print(f"[INFO] Running simulation with reward theta: {REWARD_THETA}")
        print(f"[INFO] Running simulation with reward type: {REWARD_TYPE}")
        print(f"[INFO] Running simulation with node layers: {NODE_LAYERS}")
        print(f"[INFO] Running simulation with moving average window: {MOVING_AVG_WINDOW}")
        print(f"[INFO] Running simulation with decay factor: {DECAY_FACTOR}")
    
    def change_config(self, config: ConfigManager, args:list):
        for variable in args.keys():
            value = args[variable]
            config.set(variable, value)

    def step(self, action):
        
        # calculate reward
        # record the past actions and rejections
        self.past_actions.append(action)
        current_rejection_rate = self.simulator.get_rej_rate()
        self.past_rejections.append(current_rejection_rate)
        
        if len(self.past_rejections) == self.n_steps_delay: # delay is over
            reward = self.calculate_reward(self.past_rejections)
        else:
            reward = 0  # no reward until the delay is over

        # 应用动作调整系数
        # 这里需要调用你的模拟器API或代码来更新系统状态并获取结果
        self.simulator.update_theta(action)
        
        # 模拟修改系数，并假设这导致了一些系统行为的改变
        # 例如，调整一个参数会影响拒绝率
        
        # 更新状态，这里简化为随机生成新状态
        self.state = self.simulator.get_simulator_state()
        
        # 判断是否结束回合，这里假设每次模拟运行一定时间后结束
        done = self.simulator.is_done()

        return self.state, reward, done

    def reset(self):
        # 重置环境状态为初始条件
        self.state = self.simulator.random_reset_simulator()
        # self.state = self.simulator.uniform_reset_simulator()
        return self.state

    def calculate_reward(self, past_rejections):
        reward = 0.0
        # Calculate reward based on past rejections
        for i in range(len(past_rejections)): 
            reward += past_rejections[len(past_rejections) - i - 1] * (self.decay_factor ** i)

        return -reward  

    # def render(self, mode='console'):
    #     if mode == 'console':
    #         print(f"Current state: {self.state}")
    #     else:
    #         raise NotImplementedError("Only console mode is supported")
