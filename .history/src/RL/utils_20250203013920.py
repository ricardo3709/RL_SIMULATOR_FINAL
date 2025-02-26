import numpy as np
import torch
from src.simulator.config import *
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
# device = torch.device("cpu")

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.rej_rare = np.zeros((max_size, 1))
		self.theta = np.zeros((max_size, 1))

		# priority 
		self.priorities = np.zeros((max_size, 1))
		self.alpha = 0.6  # 优先级的指数参数
		self.beta = 0.4   # 重要性采样的参数
		self.beta_increment = 0.001  # beta的增长率
		self.epsilon = 1e-6  # 避免优先级为0

		self.device = device


	def add(self, state, action, next_state, reward, done, rej_rate, theta_value):
		self.state[self.ptr] = state.flatten().cpu().detach().numpy()
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state.flatten().cpu().detach().numpy()
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.rej_rare[self.ptr] = rej_rate
		self.theta[self.ptr] = theta_value

		# 新经验的优先级设为最大优先级
		self.priorities[self.ptr] = np.max(self.priorities) if self.size > 0 else 1.0

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def get_priorities(self, indices):
		return self.priorities[indices]

	def update_priorities(self, indices, priorities):
		self.priorities[indices] = priorities + self.epsilon

	def sample(self, batch_size):
		if self.size == 0:
			return None
		
		# 简单随机采样
		indices = np.random.randint(0, self.size, size=batch_size)

		# 由于不再使用优先级采样,权重全设为1
		weights = np.ones_like(indices, dtype=np.float32)
		
		return (
			torch.FloatTensor(self.state[indices]).to(self.device),
			torch.FloatTensor(self.action[indices]).to(self.device),
			torch.FloatTensor(self.next_state[indices]).to(self.device),
			torch.FloatTensor(self.reward[indices]).to(self.device),
			torch.FloatTensor(self.not_done[indices]).to(self.device),
			torch.FloatTensor(self.rej_rare[indices]).to(self.device),
			torch.FloatTensor(self.theta[indices]).to(self.device),
			torch.FloatTensor(weights).to(self.device),
			indices
		)
	def priority_sample(self, batch_size):
		if self.size == 0:
			return None
		
		# calculater sample probability
		priorities = np.abs(self.priorities[:self.size]) + self.epsilon
		probs = priorities ** self.alpha
		probs = probs / np.sum(probs)

		# 按优先级采样
		indices = np.random.choice(self.size, batch_size, p=probs.flatten())

		# 计算重要性采样权重
		weights = (self.size * probs[indices]) ** (-self.beta)
		weights = weights / np.max(weights)
		self.beta = min(1.0, self.beta + self.beta_increment)

		# indices = np.random.randint(0, self.size, size=batch_size)

		# 返回采样的数据、权重和索引
		samples = (
			torch.FloatTensor(self.state[indices]).to(self.device),
			torch.FloatTensor(self.action[indices]).to(self.device),
			torch.FloatTensor(self.next_state[indices]).to(self.device),
			torch.FloatTensor(self.reward[indices]).to(self.device),
			torch.FloatTensor(self.not_done[indices]).to(self.device),
			torch.FloatTensor(self.rej_rare[indices]).to(self.device),
			torch.FloatTensor(self.theta[indices]).to(self.device),
			torch.FloatTensor(weights).to(self.device),
			indices
		)
	
		return samples
	
	def get_all_data(self):
        # 返回当前存储的有效样本
		return {
			'state': self.state[:self.size],
			'action': self.action[:self.size],
			'next_state': self.next_state[:self.size],
			'reward': self.reward[:self.size],
			'not_done': self.not_done[:self.size],
			'rej_rate': self.rej_rare[:self.size],
			'theta': self.theta[:self.size]
		}
	
	def merge(self, data):
		data_size = len(data['state'])
		insert_size = min(data_size, self.max_size - self.size)
		self.state[self.ptr:self.ptr + insert_size] = data['state'][:insert_size]
		self.next_state[self.ptr:self.ptr + insert_size] = data['next_state'][:insert_size]
		self.action[self.ptr:self.ptr + insert_size] = data['action'][:insert_size]
		self.reward[self.ptr:self.ptr + insert_size] = data['reward'][:insert_size]
		self.not_done[self.ptr:self.ptr + insert_size] = data['not_done'][:insert_size]
		self.rej_rare[self.ptr:self.ptr + insert_size] = data['rej_rate'][:insert_size]
		self.theta[self.ptr:self.ptr + insert_size] = data['theta'][:insert_size]
		self.ptr = (self.ptr + insert_size) % self.max_size
		self.size = min(self.size + insert_size, self.max_size)

	def get_size(self):
		return self.size
	
	def get_mean_std(self):
		states = self.get_all_data()['state']
		num_states = states.shape[0]
		reshaped_states = states.reshape(num_states, NUM_AREA, NUM_FEATURES)
		mean = np.mean(reshaped_states, axis=0)
		std = np.std(reshaped_states, axis=0)
		return mean, std
	
	def save(self, path):
		data = self.get_all_data()
		np.savez(path, **data)
	
	def load(self, path):
		data = np.load(path)

		self.__init__(data['state'].shape[1], data['action'].shape[1], data['state'].shape[0])

		self.state = data['state']
		self.action = data['action']
		self.next_state = data['next_state']
		self.reward = data['reward']
		self.not_done = data['not_done']
		self.rej_rare = data['rej_rate']
		self.theta = data['theta']
		self.size = self.state.shape[0]
		self.ptr = self.size