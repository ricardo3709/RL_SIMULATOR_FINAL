import numpy as np
import torch

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

		self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state.flatten().cpu().detach().numpy()
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state.flatten().cpu().detach().numpy()
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	
	def get_all_data(self):
        # 返回当前存储的有效样本
		return {
			'state': self.state[:self.size],
			'action': self.action[:self.size],
			'next_state': self.next_state[:self.size],
			'reward': self.reward[:self.size],
			'not_done': self.not_done[:self.size]
		}
	
	def merge(self, data):
		data_size = len(data['state'])
		insert_size = min(data_size, self.max_size - self.size)
		self.state[self.ptr:self.ptr + insert_size] = data['state'][:insert_size]
		self.next_state[self.ptr:self.ptr + insert_size] = data['next_state'][:insert_size]
		self.action[self.ptr:self.ptr + insert_size] = data['action'][:insert_size]
		self.reward[self.ptr:self.ptr + insert_size] = data['reward'][:insert_size]
		self.not_done[self.ptr:self.ptr + insert_size] = data['not_done'][:insert_size]
		self.ptr = (self.ptr + insert_size) % self.max_size
		self.size = min(self.size + insert_size, self.max_size)

	def get_size(self):
		return self.size
	
	def get_mean_std(self):
		states = self.get_all_data()['state']
		num_states = states.shape[0]
		reshaped_states = states.reshape(num_states, 63, 7)
		mean = np.mean(reshaped_states, axis=0)
		std = np.std(reshaped_states, axis=0)
		return mean, std
	
	def save(self, path):
		data = self.get_all_data()
		np.savez(path, **data)
	
	def load(self, path):
		data = np.load(path)
		self.state = data['state']
		self.action = data['action']
		self.next_state = data['next_state']
		self.reward = data['reward']
		self.not_done = data['not_done']
		self.size = self.state.shape[0]
		self.ptr = self.size