import gym
from gym import spaces
import numpy as np
from collections import deque
from src.simulator.Simulator_platform import Simulator_Platform
from src.simulator.config import *
from gym.utils import seeding
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
from tqdm import tqdm

# Check device availability: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    # print("Using CUDA device")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    # print("Using MPS device")
else:
    device = torch.device("cpu")
    # print("Using CPU device")

class ManhattanTrafficEnv(gym.Env):
    """定义曼哈顿路网模拟环境"""
    
    def __init__(self):
        super(ManhattanTrafficEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)
        
        # define the observation space, shape is number of features
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(NUM_AREA*(NUM_FEATURES),), dtype=np.float32)
        
        # initial status
        self.state = None
        self.config = ConfigManager()
        self.simulator = Simulator_Platform(0, self.config)  # Sim start time, ConfigManager

        # self.n_steps_delay = (self.config.get("RL_DURATION")/(RL_STEP_LENGTH*TIME_STEP))*2 # delay for two whole simulation
        # self.past_actions = [deque(maxlen=self.n_steps_delay)]
        self.past_actions = []
        # self.past_rejections = deque(maxlen=self.n_steps_delay)
        self.past_rejections = []
        self.past_thetas = []
        self.past_rewards = []
        self.past_anticipatory_costs = []
        self.past_operators_costs = []
        self.past_users_costs = []
        self.anticipatory_cost_sum = 0.0
        self.operators_cost_sum = 0.0
        self.users_cost_sum = 0.0

        self.step_num = 0

        self.decay_factor = self.config.get('DECAY_FACTOR')

        self.old_avg_rej = 0.0

        self.Rmax = 0.1 # maximum reward
        self.Rmin = -0.1 # minimum reward
        self.boundary_hits = 0
        # self.gnn_encoder = GNN_Encoder(num_features=NUM_FEATURES, hidden_dim=128, output_dim=32).to(device)
        # initialize the config
        self.init_config({'REWARD_THETA': 30.0, 'REWARD_TYPE': 'REJ', 'NODE_LAYERS': 2, 'MOVING_AVG_WINDOW': 40, 'DECAY_FACTOR': 1.0})

        self._max_episode_steps = 100 

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def init_config(self, args: dict):
        self.change_config(self.config, args) # change the config based on the args 
        #  = self.simulator.theta
        # REWARD_TYPE = self.config.get('REWARD_TYPE')
        # NODE_LAYERS = self.config.get('NODE_LAYERS')
        # MOVING_AVG_WINDOW = self.config.get('MOVING_AVG_WINDOW')
        # DECAY_FACTOR = self.config.get('DECAY_FACTOR')
        # SIMULATION_DURATION = self.config.get('RL_DURATION')

        # print(f"[INFO] Running simulation with reward theta: {REWARD_THETA}")
        # print(f"[INFO] Running simulation with reward type: {REWARD_TYPE}")
        # print(f"[INFO] Running simulation with node layers: {NODE_LAYERS}")
        # print(f"[INFO] Running simulation with moving average window: {MOVING_AVG_WINDOW}")
        # print(f"[INFO] Running simulation with decay factor: {DECAY_FACTOR}")
        # print(f"[INFO] Running simulation with duration: {SIMULATION_DURATION}")
    
    def change_config(self, config: ConfigManager, args:list):
        for variable in args.keys():
            value = args[variable]
            config.set(variable, value)

    def warm_up_step(self):
        current_rejection_rate = np.mean(self.simulator.current_cycle_rej_rate)
        self.past_rejections.append(current_rejection_rate)
        done = self.simulator.is_warm_up_done()
        return done

    def step(self, action):
        self.step_num += 1 # record the step number, update theta every 4 steps
        # run agent to get new_state
        next_state = self.run_sim()
        # record the past actions and rejections
        self.past_actions.append(action)
        current_rejection_rate = np.mean(self.simulator.current_cycle_rej_rate)
        self.past_rejections.append(current_rejection_rate)

        if THETA_MODE == 'STEP': #action is the change of theta
            # get new_theta and old_theta first, not update the theta
            old_theta = self.simulator.theta #[0,60]
            new_theta = old_theta + float(action)*MAX_THETA_STEP
            new_theta = np.clip(new_theta, MIN_THETA, MAX_THETA)
            # new_theta, old_theta = self.simulator.update_theta(action)
            
            if old_theta == MIN_THETA or old_theta == MAX_THETA: # if the agent is stuck in the boundary in last update
                if new_theta == old_theta: # if the agent is still stuck in the boundary
                    if self.past_rejections[-1] < REJ_THRESHOLD: # although stuck, if the rejection rate is low, give a small reward
                        reward = self.calculate_reward(self.past_rejections, action) * 0.5
                    else: # if the rejection rate is high, give a negative reward
                        self.boundary_hits += 1
                        reward = np.clip(-0.4 - self.boundary_hits*0.1, -1.0, 1.0)
                        if self.boundary_hits > 10: # if the agent is stuck in the boundary, add noise to the action
                            if old_theta == MIN_THETA:
                                new_theta = np.random.uniform(30, 60)
                            else:
                                new_theta = np.random.uniform(0, 30)
                            self.boundary_hits = 0
                            print("Reinitialize the theta to ", new_theta)
                else:
                    self.boundary_hits = 0
                    reward = self.calculate_reward(self.past_rejections, action) + 0.3 # extra reward for moving away from the boundary
            else:
                reward = self.calculate_reward(self.past_rejections, action)

            self.simulator.update_theta(new_theta) # update the theta
            self.past_thetas.append(float(new_theta))
            self.past_rewards.append(reward)

            # print(f"New Theta:{self.simulator.theta}")

        elif THETA_MODE == 'DIRECT': #action is the direct new theta
            new_theta = float(action) * MAX_THETA/2 + MAX_THETA/2 # scale the action to theta 
            self.simulator.update_theta(new_theta) # update the theta
            reward = self.calculate_reward(self.past_rejections)
        else:
            raise ValueError(f"Theta mode {THETA_MODE} is not supported")
        
        if ENV_MODE != 'TEST':    
            print(f"Reward: {reward}")
            print(f"current rejection rate: {current_rejection_rate}")
            print(f"current user cost: {self.anticipatory_cost_sum}")
        
        done = self.simulator.is_done()

        # should return next_state, reward, done, _ 
        return next_state, reward, done, new_theta
    
    def get_state_tensor(self):
        next_x= self.simulator.get_simulator_state_by_areas()

        # rej_tensor = torch.tensor([0.0], dtype=torch.float32).to(device).unsqueeze(0).expand(next_x.size(0), -1)
        # theta_tensor = torch.tensor([30.0], dtype=torch.float32).to(device).unsqueeze(0).expand(next_x.size(0), -1)
        # next_x = torch.cat((next_x, rej_tensor, theta_tensor), dim=1)

        return next_x # [NUM_AREA,NUM_FEATURES]

    def reset(self):
        # random reset simulator
        self.simulator.random_reset_simulator()
        # self.simulator.uniform_reset_simulator()
        next_x = self.simulator.get_simulator_state_by_areas()
        
        # network = self.simulator.get_simulator_network_by_areas()
        # edge_index = self.graph_to_data(network)

        # rej_tensor = torch.tensor([0.0], dtype=torch.float32).to(device).unsqueeze(0).expand(next_x.size(0), -1)
        # theta_tensor = torch.tensor([30.0], dtype=torch.float32).to(device).unsqueeze(0).expand(next_x.size(0), -1)
        # next_x = torch.cat((next_x, rej_tensor, theta_tensor), dim=1)
        # next_graph_data = Data(x=next_x, edge_index=edge_index)

        # next_state_encoded = self.gnn_encoder(next_graph_data)  # encode the state (1,32)
        return next_x # [NUM_AREA,NUM_FEATURES]
    
    def uniform_reset(self):
        self.__init__()
        print(f"Uniform reset the environment")
        self.simulator.uniform_reset_simulator()
        print(f"mean rej{np.mean(self.past_rejections)}")
        next_x = self.simulator.get_simulator_state_by_areas()
        rej_tensor = torch.tensor([0.0], dtype=torch.float32).to(device).unsqueeze(0).expand(next_x.size(0), -1)
        theta_tensor = torch.tensor([30.0], dtype=torch.float32).to(device).unsqueeze(0).expand(next_x.size(0), -1)
        next_x = torch.cat((next_x, rej_tensor, theta_tensor), dim=1)
        return next_x # [NUM_AREA,7]

     
    def calculate_reward(self, past_rejections, action = 0.0):
        # if len(past_rejections) > 2:
        if False:
            recent_rejections = past_rejections[-PAST_REJ_NUM:]
            mean_rejection = np.mean(recent_rejections)

            rej_reward = (0.6 - mean_rejection) * 3
            rej_reward = max(-1.0, min(1.0, rej_reward)) #clip the reward to [-0.5, 0.5]
            anticipatory_reward = - self.anticipatory_cost_sum * 0.02
            reward = rej_reward + anticipatory_reward
            
            if ENV_MODE != 'TEST':
                print(f"Average rejection rate (last {PAST_REJ_NUM}): {mean_rejection:.4f}")
                print(f"Anticipatory cost: {anticipatory_reward:.4f}")
                print(f"Rejection rate reward: {rej_reward:.4f}")
        else:
            recent_rej = past_rejections[-PAST_REJ_NUM:]
            mean_rej = np.mean(recent_rej)
            reward_rej = (REJ_THRESHOLD_REWARD - mean_rej) * REJ_THRESHOLD_MULTIPLIER
            reward_cost = (BASE_USER_COST - self.anticipatory_cost_sum) * COST_REWARD_MULTIPLIER

            reward = reward_rej + reward_cost + REWARD_OFFSET # add offset to the reward to avoid too many negative reward
            # reward = max(-1.0, min(1.0, reward*REWARD_MULTIPLIER)) #clip the scaled reward to [-1.0, 1.0]

            # # try simple reward first
            # rej_reward = (REJ_THRESHOLD_REWARD - past_rejections[-1]) * REJ_THRESHOLD_MULTIPLIER
            # rej_reward = max(-1.0, min(1.0, rej_reward)) #clip the reward to [-0.5, 0.5]
            # if COST_TYPE == 'USER':
            #     extra_cost = self.anticipatory_cost_sum - BASE_USER_COST
            # cost_reward = - extra_cost * COST_REWARD_MULTIPLIER
            # # anticipatory_reward = - self.anticipatory_cost_sum * ANTICIAPATORY_REWARD_MULTIPLIER
            # reward = rej_reward + cost_reward
            # if ENV_MODE != 'TEST':
            #     print(f"Cost Reward: {cost_reward}")
            #     print(f"Rej Reward:{rej_reward}")
            
        return reward
        # if len(past_rejections) < 2:
        #     return 0.0
        # elif past_rejections[-1] < REJ_THRESHOLD:
        #     extra_reward = (REJ_THRESHOLD - past_rejections[-1]) * 10
        #     return extra_reward + 1.0
        # else:
        #     origin_reward = past_rejections[-2] - past_rejections[-1]
        #     normalized_reward = 2 * ((origin_reward - self.Rmin) / (self.Rmax - self.Rmin)) - 1
        # return normalized_reward
    
    def calculate_reward_ori(self, past_rejections):
        LEARNING_WINDOW = self.config.get('LEARNING_WINDOW')
        CONSIDER_NUM_CYCLES = self.config.get('CONSIDER_NUM_CYCLES')
        CYCLE_WINDOW = int(LEARNING_WINDOW/(RL_STEP_LENGTH*TIME_STEP))

        cycle_reward = 0.0
        old_avg_rej = np.mean(past_rejections[-CYCLE_WINDOW*2 : -CYCLE_WINDOW]) # last 60mins to last 30mins
        current_avg_rej = np.mean(past_rejections[-CYCLE_WINDOW:]) # last 30mins
        current_cycle_rej = past_rejections[-1]

        if current_cycle_rej > 1: #bug handle
            return 0.0

        cycle_weight = 0.9
        long_weight = 0.1

        for cycle in range(1, CONSIDER_NUM_CYCLES+1):
            last_cycle_rej = past_rejections[-(cycle+1)]
            cycle_reward += (last_cycle_rej - current_cycle_rej) * (self.decay_factor ** cycle)
        # # Calculate reward based on past rejections
        # for i in range(len(past_rejections)-1): 
        #     old_avg_rej += past_rejections[i] * (self.decay_factor ** i) 
        # reward = old_avg_rej - current_rej # reward is positive if the current rejection rate is lower than the past average
        long_reward = (old_avg_rej - current_avg_rej) 
        combined_reward = cycle_weight * cycle_reward + long_weight * long_reward
        normalized_reward = 2 * ((combined_reward - self.Rmin) / (self.Rmax - self.Rmin)) - 1
        return normalized_reward

    def graph_to_data(self, network):
        # Convert the network to edge_index
        if isinstance(network, pd.DataFrame):
            network_array = network.values
        else:
            network_array = network

        # Find the indices of non-zero elements, which correspond to edges
        src_nodes, dst_nodes = np.nonzero(network_array)
        # edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

        # 合并成一个 numpy 数组
        edges = np.vstack((src_nodes, dst_nodes))

        # 转换为 PyTorch 张量
        edge_index = torch.tensor(edges, dtype=torch.long)

        # edge_index = edge_index.to(device)
        # print(edge_index)
        
        # Ensure the result is in the correct shape (2, num_edges)
        return edge_index

    def run_sim(self):
        current_sim_time = self.simulator.system_time
        last_req_ID = self.simulator.last_req_ID
        if MAP_NAME == 'Manhattan':
            if current_sim_time % (3600*8 == 0 and current_sim_time > 0:
                self.simulator.req_loader(current_sim_time, last_req_ID)
        elif MAP_NAME == 'Utrecht':
            if current_sim_time % 3600*2 == 0 and current_sim_time > 0:
                self.simulator.req_loader(current_sim_time, last_req_ID)
                
        # if current_sim_time % 3600 == 0 and current_sim_time > 0: #end of the hour
        #     self.simulator.req_loader(current_sim_time, last_req_ID)
        
        # network = self.simulator.get_simulator_network_by_areas()

        # edge_index = self.graph_to_data(network)
        # anticipatory_cost_sum = 0.0
        operators_cost_sum = 0.0
        users_cost_sum = 0.0

        self.simulator.current_cycle_rej_rate = [] # reset the rejection rate to record the new cycle
        if ENV_MODE == 'TEST':
            for step in range(int(RL_STEP_LENGTH)): # 2.5mins, 10 steps
                self.simulator.system_time += TIME_STEP
                step_operator_cost, step_user_cost = self.simulator.run_cycle() # Run one cycle(15s)
                # anticipatory_cost_sum += step_anticipatory_cost
                self.past_operators_costs.append(step_operator_cost)
                self.past_users_costs.append(step_user_cost)
                operators_cost_sum += step_operator_cost
                users_cost_sum += step_user_cost
                # self.past_anticipatory_costs.append(step_anticipatory_cost)
                
        else: 
            for step in tqdm(range(int(RL_STEP_LENGTH)), desc=f'Run SIM'): # 2.5mins, 10 steps
                self.simulator.system_time += TIME_STEP
                # step_anticipatory_cost = self.simulator.run_cycle() # Run one cycle(15s)
                # anticipatory_cost_sum += step_anticipatory_cost
                # self.past_anticipatory_costs.append(step_anticipatory_cost)
                step_operator_cost, step_user_cost = self.simulator.run_cycle() # Run one cycle(15s)
                self.past_operators_costs.append(step_operator_cost)
                self.past_users_costs.append(step_user_cost)
                operators_cost_sum += step_operator_cost
                users_cost_sum += step_user_cost
                # print(f'Step User Cost: {step_user_cost}')
        
        if COST_TYPE == 'OPERATOR':
            anticipatory_cost_sum = operators_cost_sum
        elif COST_TYPE == 'USER':
            anticipatory_cost_sum = users_cost_sum
            # print(f'User cost sum: {users_cost_sum}')
        elif COST_TYPE == 'BOTH':
            anticipatory_cost_sum = operators_cost_sum + users_cost_sum
        else:
            raise ValueError(f"Cost type {COST_TYPE} is not supported")
        
        avg_anticipatory_cost = anticipatory_cost_sum / int(RL_STEP_LENGTH)
        self.anticipatory_cost_sum = avg_anticipatory_cost

        next_state = self.simulator.get_simulator_state_by_areas()
        # rej_rate = np.mean(self.simulator.current_cycle_rej_rate)
        # state_theta = self.simulator.theta

        next_state = next_state.clone().detach().requires_grad_(True)
        next_state = next_state.to(device) # [NUM_AREA,NUM_FEATURES]

        # rej_tensor = torch.tensor([rej_rate], dtype=torch.float32).to(device).unsqueeze(0).expand(next_state.size(0), -1)
        # theta_tensor = torch.tensor([state_theta], dtype=torch.float32).to(device).unsqueeze(0).expand(next_state.size(0), -1)
        # theta_tensor = (theta_tensor - MAX_THETA/2)/(MAX_THETA/2)  # scale the theta to [-1,1]
        # next_state = torch.cat((next_state, rej_tensor, theta_tensor), dim=1) #shape: [NUM_AREA,7]
        return next_state



# class GNN_Encoder(nn.Module):
#     def __init__(self, num_features=NUM_FEATURES, hidden_dim=64, output_dim=32):
#         super(GNN_Encoder, self).__init__()
#         self.conv1 = GCNConv(num_features, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.conv3 = GCNConv(hidden_dim, output_dim)
#         self.ln1 = nn.LayerNorm(output_dim+NUM_FEATURES)
#         # self.normalizer = Normalizer(size=output_dim+NUM_FEATURES)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, GCNConv):
#             # 初始化 GCNConv 的权重
#             for param in m.parameters():
#                 if param.data.dim() > 1:
#                     nn.init.xavier_uniform_(param.data, gain=nn.init.calculate_gain('relu'))
#                 else:
#                     nn.init.zeros_(param.data)
#         elif isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.ones_(m.weight)
#             nn.init.zeros_(m.bias)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = x.float().to(device)
#         origin_graph = x.clone()
#         edge_index = edge_index.to(device)

#         x = torch.relu(self.conv1(x, edge_index))
#         x = torch.relu(self.conv3(x, edge_index))
#         x = torch.cat((x, origin_graph), dim=1)
       
#         # Normalize output features
#         # for i in range(x.size(0)):
#         #     self.normalizer.observe(x[i, :])
#         # x = torch.stack([self.normalizer.normalize(x[i, :]) for i in range(x.size(0))])

#         x = self.ln1(x)
#         return x
