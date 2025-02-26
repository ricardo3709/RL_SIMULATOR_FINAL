import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import logging
from torch_geometric.nn import GCNConv, global_mean_pool
from src.simulator.config import *
from copy import deepcopy
from torch_geometric.utils import to_dense_adj
# from torch_geometric.loader import DataLoader
# from torch_geometric.data import Data

# Check device availability: CUDA > MPS > CPU
if torch.cuda.is_available():
	device = torch.device("cuda")
	print("Using CUDA device")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
	device = torch.device("mps")
	print("Using MPS device")
else:
	device = torch.device("cpu")
	print("Using CPU device")

actor_lr = 1e-5
critic_lr = 1e-4
# gnn_lr = 1e-3

class GNN_Encoder(nn.Module):
	def __init__(self, num_features=NUM_FEATURES, hidden_dim=8, output_dim=256):
		super(GNN_Encoder, self).__init__()

		self.conv1 = GCNConv(num_features, hidden_dim)
		self.conv2 = GCNConv(hidden_dim, hidden_dim*2)
		
		self.fc = nn.Linear(hidden_dim*2 + NUM_FEATURES + 2, int(output_dim/NUM_AREA))
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

	def _init_weights(self, m):
		if isinstance(m, GCNConv):
			# 初始化 GCNConv 的权重
			for param in m.parameters():
				if param.data.dim() > 1:
					nn.init.xavier_uniform_(param.data, gain=nn.init.calculate_gain('leaky_relu'))
				else:
					nn.init.zeros_(param.data)
		elif isinstance(m, nn.Linear):
			nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
			if m.bias is not None:
				nn.init.zeros_(m.bias)
		elif isinstance(m, nn.LayerNorm):
			nn.init.ones_(m.weight)
			nn.init.zeros_(m.bias)

	def forward(self, state, edge_index, batch_size, rej_rate, theta_value):
		if SKIP_GNN:
			return state
		
		edge_index = edge_index.to(device) # [NUM_AREA, NUM_AREA]
		x = state.to(device) # [b, NUM_AREA, NUM_FEATURES] 
		x = x.reshape(batch_size, NUM_AREA, NUM_FEATURES) # [b, NUM_AREA, NUM_FEATURES]
		origin_graph = x.clone()

		x = self.leaky_relu(self.conv1(x, edge_index)) #[b, NUM_AREA, hidden_dim]
		x = self.leaky_relu(self.conv2(x, edge_index)) #[b, NUM_AREA, hidden_dim*2]
		x = torch.cat((x, origin_graph), dim=2)  #[b, NUM_AREA, hidden_dim*2 + NUM_FEATURES]

		# Cat Rejection Rate and Theta Value
		rej_rate = rej_rate.squeeze() 
		theta_value = theta_value.squeeze()
		# print(f"rej_rate: {rej_rate}")
		# print(f"theta_value: {theta_value}")
		if not isinstance(rej_rate, torch.Tensor):
			rej_rate = torch.tensor(rej_rate, dtype=torch.float32).to(device)
		if not isinstance(theta_value, torch.Tensor):
			theta_value = torch.tensor(theta_value, dtype=torch.float32).to(device)
		
		if rej_rate.dim() == 0: # in eval, the rej_rate is a scalar
			rej_rate = rej_rate.unsqueeze(0)
		if theta_value.dim() == 0:
			theta_value = theta_value.unsqueeze(0)
		
		global_features = torch.stack([rej_rate, theta_value], dim=1)  # [batch_size, 2]
		global_features = global_features.unsqueeze(1).expand(-1, NUM_AREA, -1)  # [batch_size, NUM_AREA, 2]

		x = torch.cat([x, global_features], dim=2) # [b, NUM_AREA, hidden_dim*2 + NUM_FEATURES + 2]

		# 降维
		x = self.fc(x) # [b, NUM_AREA, output_dim/NUM_AREA]
		# Flatten
		x = x.view(batch_size, -1) # [b, output_dim]
		
		return F.tanh(x)



		if state.shape[0] == 63:																						
			x = state.unsqueeze(0).to(device)
		else:
			x = state.reshape(batch_size,63,NUM_FEATURES).to(device) #[256,63,7]
		# x = state.to(device)
		edge_index = edge_index.to(device)
		# x = x.float().to(device)
		origin_graph = x.clone()

		x = torch.relu(self.conv1(x, edge_index)) #[b, 63, 64]
		x = torch.relu(self.conv2(x, edge_index)) #[b, 63, 32]
		x = torch.cat((x, origin_graph), dim=2) #[b, 63, 39]
		
		x = x.view(batch_size, -1) # [b, 63*39]
		# print(f"GNN L1 Weights: {self.conv1.lin.weight}")
		# print(f"GNN L2 Weights: {self.conv2.lin.weight}")
		# print(f"GNN Output: {x}")
		return x
	
# class GNN_AutoEncoder(nn.Module):
# 	def __init__(self, num_features=NUM_FEATURES, hidden_dim=256, output_dim=NUM_AREA*2):
# 		super(GNN_AutoEncoder, self).__init__() 

# 		self.encoder = GNN_Encoder(num_features=num_features, hidden_dim=hidden_dim, output_dim=output_dim)
# 		self.decoder = GNN_Decoder(output_dim=output_dim, hidden_dim=hidden_dim, num_features=num_features)
# 		# self.reward_predictor = nn.Linear(output_dim, 1)

# 	def forward(self, x, edge_index, batch_size, rej_rate, theta_value):
# 		encoded, node_embeddings = self.encoder(x, edge_index, batch_size, rej_rate, theta_value)
# 		decoded = self.decoder(encoded, edge_index, batch_size) 
# 		adj_hat = self.reconstruct_adj(node_embeddings)

# 		return encoded, decoded, adj_hat
	
# 	# def reconstruct_adj(self, encoded):
# 	# 	batch_size = encoded.size(0)
# 	# 	adj_flat = self.adj_reconstructor(encoded)
# 	# 	adj = adj_flat.view(batch_size, self.num_nodes, self.num_nodes)
		
# 	# 	# 确保邻接矩阵是对称的
# 	# 	adj = (adj + adj.transpose(1, 2)) / 2
		
# 	# 	return adj
# 	def reconstruct_adj(self, node_embeddings):
# 		if SKIP_GNN:
# 			return None
# 		# z = z.view(z.size(0), NUM_AREA, NUM_FEATURES)  # [batch_size, num_nodes, output_dim]
# 		# adj_hat = torch.sigmoid(torch.matmul(z, z.transpose(1, 2)))  # [batch_size, num_nodes, num_nodes]
# 		adj_hat = torch.sigmoid(torch.matmul(node_embeddings, node_embeddings.transpose(1,2)))  # [batch_size, num_nodes, num_nodes]
# 		return adj_hat
	
# class GNN_Decoder(nn.Module):
# 	def __init__(self, output_dim, hidden_dim=64, num_features=NUM_FEATURES):
# 		super(GNN_Decoder, self).__init__()

# 		# 全连接层，将图表示映射回节点特征
# 		self.fc1 = nn.Linear(output_dim, hidden_dim * 2)
# 		# self.ln1 = nn.LayerNorm(hidden_dim * 2)

# 		self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim * NUM_AREA)
# 		# self.ln2 = nn.LayerNorm(hidden_dim * NUM_AREA)
		
# 		# 将全连接层的输出重塑为节点特征
# 		self.node_feature_dim = hidden_dim  # 重建的节点特征维度

# 		# 图卷积层，细化节点特征
# 		self.conv1 = GCNConv(hidden_dim, hidden_dim // 2)
# 		self.res1 = nn.Linear(hidden_dim, hidden_dim // 2)

# 		self.conv2 = GCNConv(hidden_dim // 2, num_features)  # 输出节点的原始特征维度
# 		self.res2 = nn.Linear(hidden_dim // 2, num_features)

# 		self.apply(self._init_weights)
	
# 	def _init_weights(self, m):
# 		if isinstance(m, nn.Linear):
# 			if m in [self.res1, self.res2]:  # 残差连接的初始化
# 				# 接近于单位映射
# 				nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='linear')
# 				nn.init.constant_(m.bias, 0)
# 				# 缩小初始权重，让残差路径的贡献更小
# 				m.weight.data *= 0.1
# 			else:
# 				# 其他线性层使用原来的初始化
# 				nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
# 				nn.init.constant_(m.bias, 0.01)
# 		elif isinstance(m, GCNConv):
# 			nn.init.xavier_uniform_(m.lin.weight, gain=0.1)
# 			if m.bias is not None:
# 				nn.init.constant_(m.bias, 0)

# 	def forward(self, x, edge_index, batch_size):
# 		if SKIP_GNN:  # 跳过 GNN
# 			return x
# 		# x 的形状为 [batch_size, output_dim]

# 		edge_index = edge_index.to(device)
# 		x = x.to(device)

# 		# 全连接层映射
# 		x = F.leaky_relu(self.fc1(x))  # [batch_size, hidden_dim * 2]
# 		# x = self.ln1(x)

# 		x = F.leaky_relu(self.fc2(x))          # [batch_size, hidden_dim * NUM_AREA]
# 		# x = self.ln2(x)

# 		# 重塑为 [batch_size * NUM_AREA, hidden_dim]
# 		x = x.view(batch_size * NUM_AREA, self.node_feature_dim)
		
# 		# 第一个残差块
# 		identity1 = self.res1(x)
# 		# out1 = F.leaky_relu(self.conv1(x, edge_index))
# 		out1 = self.conv1(x, edge_index)
# 		x = out1 + identity1
# 		x = F.leaky_relu(x)

# 		# 第二个残差块
# 		identity2 = self.res2(x)
# 		out2 = self.conv2(x, edge_index)
# 		x = out2 + identity2

# 		# 如果需要，将节点特征重塑为 [batch_size, NUM_AREA, num_features]
# 		x = x.view(batch_size, NUM_AREA, NUM_FEATURES)

# 		return x  # 返回重建的节点特征
	
# class GNN_Encoder(nn.Module):
# 	def __init__(self, num_features=NUM_FEATURES, hidden_dim=128, output_dim=NUM_AREA*1):
# 		super(GNN_Encoder, self).__init__()

# 		# self.conv1 = GCNConv(num_features, hidden_dim)
# 		# self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)

# 		# self.fc1 = nn.Linear(NUM_AREA * hidden_dim * 2, hidden_dim * 4)
# 		# self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
# 		# self.fc3 = nn.Linear(hidden_dim * 2, output_dim)

# 		# self.bn1 = nn.BatchNorm1d(hidden_dim)
# 		# self.bn2 = nn.BatchNorm1d(hidden_dim * 2)

# 		self.conv1 = GCNConv(num_features, hidden_dim)
# 		self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)

# 		# self.bn1 = nn.BatchNorm1d(hidden_dim)
# 		# self.bn2 = nn.BatchNorm1d(hidden_dim * 2)

# 		# Residual connection
# 		self.res1 = nn.Linear(num_features, hidden_dim)
# 		self.res2 = nn.Linear(hidden_dim, hidden_dim * 2)

# 		# 更新全连接层的输入维度
# 		# 假设 embedding_dim = hidden_dim * 2
# 		embedding_dim = hidden_dim * 2
# 		self.fc1 = nn.Linear(embedding_dim + 2, 256)  # 加上全局特征的维度 2
# 		# self.ln1 = nn.LayerNorm(256)

# 		self.fc2 = nn.Linear(256, 128)
# 		# self.ln2 = nn.LayerNorm(128)

# 		self.fc3 = nn.Linear(128, output_dim)

# 		self.apply(self._init_weights)
	
# 	def _init_weights(self, m):
# 		if isinstance(m, nn.Linear):
# 			# 对最后一层（使用tanh的层）使用较小的初始化范围
# 			if m == self.fc3:  # 假设fc3是最后一层
# 				# xavier初始化对于tanh来说是个好选择
# 				nn.init.xavier_uniform_(m.weight, gain=0.1)
# 				nn.init.uniform_(m.bias, -0.1, 0.1)
# 			else:
# 				if m in [self.res1, self.res2]:  # 残差连接的初始化
# 					# 接近于单位映射
# 					nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='linear')
# 					nn.init.constant_(m.bias, 0)
# 					# 缩小初始权重，让残差路径的贡献更小
# 					m.weight.data *= 0.1
# 				else:
# 					# 其他线性层使用原来的初始化
# 					nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
# 					nn.init.constant_(m.bias, 0.01)
# 		elif isinstance(m, GCNConv):
# 			nn.init.xavier_uniform_(m.lin.weight, gain=0.1)
# 			if m.bias is not None:
# 				nn.init.constant_(m.bias, 0)

# 	def forward(self, x, edge_index, batch_size, rej_rate, theta_value):
# 		if SKIP_GNN:
# 			return x, None
# 		# 确保输入维度正确
# 		if x.dim() == 2:
# 			x = x.unsqueeze(0) #[1, batch_size, NUM_AREA*NUM_FEATURES]
# 		x = x.view(batch_size*NUM_AREA, NUM_FEATURES).to(device)  # [batch_size*NUM_AREA, num_features]

# 		# # edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
# 		# edge_index = edge_index.to(device)
# 		# edge_index_batch_list = []
# 		# for i in range(batch_size):
# 		# 	area_offset = i * NUM_AREA
# 		# 	edge_index_i = edge_index + area_offset
# 		# 	edge_index_batch_list.append(edge_index_i)
# 		# edge_index = torch.cat(edge_index_batch_list, dim=1) # [2, 10240] each graph contains 160 edges, 32 areas, 5 features

# 		edge_index = edge_index.to(device)
# 		batch = torch.arange(batch_size).unsqueeze(1).repeat(1, NUM_AREA).view(-1).to(device) #[2048] NUM_AREA*batch_size

# 		# 检查 edge_index 中的节点索引是否超出范围
# 		max_node_index = x.size(0)
# 		if edge_index.max() >= max_node_index or edge_index.min() < 0:
# 			print("Error: edge_index contains invalid node indices.")
# 			print("edge_index_batch.max() =", edge_index.max())
# 			print("max_node_index =", max_node_index)
# 			# 可以在这里选择抛出异常或进行其他处理
# 			raise ValueError("edge_index contains invalid node indices.")
		
# 		identity1 = self.res1(x)
# 		# out1 = F.leaky_relu(self.conv1(x, edge_index), negative_slope=0.1)
# 		out1 = self.conv1(x, edge_index)
# 		x = out1 + identity1  # 残差连接
# 		x = F.leaky_relu(x, negative_slope=0.1)

# 		identity2 = self.res2(x)
# 		# out2 = F.leaky_relu(self.conv2(x, edge_index), negative_slope=0.1)
# 		out2 = self.conv2(x, edge_index)
# 		x = out2 + identity2  # 残差连接
# 		x = F.leaky_relu(x, negative_slope=0.1) # [batch_size * NUM_AREA, hidden_dim * 2]

# 		#get node embeddings
# 		node_embeddings = x.view(batch_size, NUM_AREA, -1) # [batch_size, NUM_AREA, hidden_dim * 2]

# 		#global mean pool
# 		graph_embedding = global_mean_pool(x, batch) # [batch_size, embedding_dim]

# 		#cat global embedding with rejection rate and theta value
# 		rej_rate = rej_rate.squeeze() 
# 		theta_value = theta_value.squeeze()
# 		if not isinstance(rej_rate, torch.Tensor):
# 			rej_rate = torch.tensor(rej_rate, dtype=torch.float32).to(device)
# 		if not isinstance(theta_value, torch.Tensor):
# 			theta_value = torch.tensor(theta_value, dtype=torch.float32).to(device)
		
# 		if rej_rate.dim() == 0: # in eval, the rej_rate is a scalar
# 			rej_rate = rej_rate.unsqueeze(0)
# 		if theta_value.dim() == 0:
# 			theta_value = theta_value.unsqueeze(0)

# 		global_features = torch.stack([rej_rate, theta_value], dim=1).to(device)  # [batch_size, 2]
# 		# print("graph_embedding: ", graph_embedding.size())
# 		# print("global_features: ", global_features.size())
# 		# print("rej_rate: ", rej_rate.size())
# 		# print("theta_value: ", theta_value.size())
# 		x = torch.cat([graph_embedding, global_features], dim=1) # [batch_size, embedding_dim + 2]

# 		#fc layers
# 		x = F.leaky_relu(self.fc1(x), negative_slope=0.01) # [batch_size, 256]
# 		# x = self.ln1(x)

# 		x = F.leaky_relu(self.fc2(x), negative_slope=0.01) # [batch_size, 128]
# 		# x = self.ln2(x)

# 		x = F.tanh(self.fc3(x)) # [batch_size, output_dim]
		
# 		return x, node_embeddings
		
# 		# 应用图卷积
# 		x = F.relu(self.bn1(self.conv1(x, edge_index).transpose(1, 2)).transpose(1, 2))  # [batch_size, NUM_AREA, 64]
# 		x = F.relu(self.bn2(self.conv2(x, edge_index).transpose(1, 2)).transpose(1, 2))  # [batch_size, NUM_AREA, 128]
		
# 		# 展平
# 		# x = x.view(batch_size, -1)  # [batch_size, NUM_AREA * 128]
# 		x = x.reshape(batch_size, -1)
		
# 		# 全连接层
# 		x = F.relu(self.fc1(x))  # [batch_size, 256]
# 		x = F.relu(self.fc2(x))  # [batch_size, 128]
# 		x = self.fc3(x)  # [batch_size, 512]
		
# 		return x

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		# # old version with normalization
		# self.l1 = nn.Linear(state_dim, 512)
		# self.l2 = nn.Linear(512, 256)
		# self.l3 = nn.Linear(256, 128)
		# self.l4 = nn.Linear(128, action_dim)

		# # ADD layer normalization
		# self.ln1 = nn.LayerNorm(512)
		# self.ln2 = nn.LayerNorm(256)
		# self.ln3 = nn.LayerNorm(128)

		# # # 添加动作约束层
		# # self.action_scale = nn.Parameter(torch.ones(1) * 0.1)
		# self.max_action = max_action
		# self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

		self.l1 = nn.Linear(state_dim, 128)
		self.l2 = nn.Linear(128, 64)
		self.l3 = nn.Linear(64, 32)
		self.l4 = nn.Linear(32, action_dim)
		# self.l4 = nn.Linear(32, 16)
		# self.l5 = nn.Linear(16, action_dim)

		# # Simple Version
		# self.l1 = nn.Linear(state_dim, 16)
		# self.l2 = nn.Linear(16, action_dim)
		
		# 只保留ReLU激活函数
		self.relu = nn.ReLU()
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

		self.logger = TD3Logger()
		
		self.max_action = max_action
		self.apply(self._init_weights)

		self.grad_stats = {}
		self.activation_stats = {}

		# def log_gradients(self):
		# 	"""记录每一层的梯度"""
		# 	for name, param in self.named_parameters():
		# 		if param.grad is not None:
		# 			self.grad_stats[f"param_{name}"] = param.grad.norm().item()
				
		# def log_activations(self, activations):
		# 	"""记录每一层的激活值统计"""
		# 	for name, act in activations.items():
		# 		self.activation_stats[name] = {
		# 			'mean': act.mean().item(),
		# 			'std': act.std().item(),
		# 			'max': act.max().item(),
		# 			'min': act.min().item()
		# 		}
		
	def _init_weights(self, m):
			# if isinstance(m, nn.Linear):
			# 	if m == self.l4:
			# 		nn.init.uniform_(m.weight, -0.003, 0.003)
			# 		nn.init.uniform_(m.bias, -0.003, 0.003)
			# 	else:
			# 		# Kaiming 初始化，考虑 LeakyReLU 的负斜率
			# 		nn.init.kaiming_normal_(m.weight, 
			# 							a=0.01,  # LeakyReLU 的负斜率
			# 							mode='fan_in', 
			# 							nonlinearity='leaky_relu')
			# 		if m.bias is not None:
			# 			nn.init.zeros_(m.bias)
			
			if isinstance(m, nn.Linear):
				if m == self.l4:
					nn.init.uniform_(m.weight, -0.05, 0.05)
					nn.init.uniform_(m.bias, -0.05, 0.05)
				else:
					# kaiming init, with relu
					# nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
					nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
					if m.bias is not None:
						nn.init.constant_(m.bias, 0.01)
			elif isinstance(m, nn.LayerNorm):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, state):
		# a = self.ln1(self.leaky_relu(self.l1(state)))
		# b = self.ln2(self.leaky_relu(self.l2(a)))
		# c = self.ln3(self.leaky_relu(self.l3(b)))

		# # 只在训练时进行激活值统计和噪声添加
		# if self.training:
		# 	with torch.no_grad():
		# 		c_saturated = (torch.abs(c) > 0.9).float().mean().item()
				
		# 		# 根据饱和程度动态调整噪声强度
		# 		if c_saturated > 0.8:
		# 			noise_scale = 0.1 * min(c_saturated, 1.0)
		# 			print(f"Warning: Actor pre-output saturated: {c_saturated:.2f}, adding noise with scale {noise_scale:.3f}")
		# 			c = c + torch.randn_like(c) * noise_scale
		# 		else:
		# 			# 始终添加小噪声以维持探索
		# 			c = c + torch.randn_like(c) * 0.01

		# New Version
		if SKIP_GNN:
			state = state.reshape(-1, NUM_AREA * NUM_FEATURES).to(device)

		# a = self.leaky_relu(self.l1(state))
		a = self.l1(state)
		# a.retain_grad()
		self.activation_stats['layer1'] = a
		a = self.leaky_relu(a)
		# a = self.relu(self.l1(state))
		# a = self.ln1(a)

		# b = self.leaky_relu(self.l2(a))
		b = self.l2(a)
		# b.retain_grad()
		self.activation_stats['layer2'] = b
		b = self.leaky_relu(b)
		# b = self.relu(self.l2(a))
		# b = self.ln2(b)

		# c = self.leaky_relu(self.l3(b))
		c = self.l3(b)
		# c.retain_grad()
		self.activation_stats['layer3'] = c
		c = self.leaky_relu(c)
		# c = self.relu(self.l3(b))
		# c = self.ln3(c)
		
		output = self.l4(c)
		self.activation_stats['output'] = output
		# d = self.l4(c)
		# activations['layer4'] = d
		# d = self.leaky_relu(d)

		# output = self.l5(d)
		# activations['output'] = output
		# d.retain_grad()
		
		# # 3. 记录日志
		# if self.training and hasattr(self, 'grad_stats'):
		# 	self.logger.log_gradient_stats(self.grad_stats, self.forward_count)

		# # 收集激活统计（每n次迭代打印一次，避免刷屏）
		# if hasattr(self, 'forward_count'):
		# 	self.forward_count += 1
		# else:
		# 	self.forward_count = 0
			
		# if self.forward_count % 100 == 0:  # 每100次前向传播打印一次
		# 	activation_stats = {
		# 		'layer1': {'mean': a.mean().item(), 
		# 					'std': a.std(unbiased=False).item() if a.numel() > 1 else 0.0, 
		# 					'saturated': (torch.abs(a) > 0.9).float().mean().item()},
		# 		'layer2': {'mean': b.mean().item(), 
		# 					'std': b.std(unbiased=False).item() if a.numel() > 1 else 0.0,
		# 					'saturated': (torch.abs(b) > 0.9).float().mean().item()},
		# 		'layer3': {'mean': c.mean().item(), 
		# 					'std': c.std(unbiased=False).item() if a.numel() > 1 else 0.0,
		# 					'saturated': (torch.abs(c) > 0.9).float().mean().item()},
		# 		'layer4': {'mean': d.mean().item(),
		# 					'std': d.std(unbiased=False).item() if a.numel() > 1 else 0.0,
		# 					'saturated': (torch.abs(d) > 0.9).float().mean().item()},
		# 		# 'layer5': {'mean': e.mean().item(),
		# 		# 			'std': e.std(unbiased=False).item() if a.numel() > 1 else 0.0,
		# 		# 			'saturated': (torch.abs(e) > 0.9).float().mean().item()},
		# 		'output': {'mean': output.mean().item(),	
		# 					'std': output.std(unbiased=False).item() if a.numel() > 1 else 0.0,
		# 					'saturated': (torch.abs(output) > 0.9).float().mean().item()}

		# 	}
		# 	print("Activation stats:", activation_stats)
		# 	self.logger.log_activation_stats(activation_stats, self.forward_count)
		
		        
		# # 只在需要记录梯度时设置hooks
		# if self.training and hasattr(self, 'grad_stats'):
		# 	def assign_grad_stats(layer, grad):
		# 		if grad is not None:  # 确保梯度存在
		# 			self.grad_stats[layer] = grad.norm().item()

		# 	# 使用autograd.Function来记录梯度
		# 	class GradientRecorder(torch.autograd.Function):
		# 		@staticmethod
		# 		def forward(ctx, input, name):
		# 			ctx.name = name
		# 			return input
				
		# 		@staticmethod
		# 		def backward(ctx, grad_output):
		# 			assign_grad_stats(ctx.name, grad_output)
		# 			return grad_output, None

		# 	a = GradientRecorder.apply(a, 'layer1')
		# 	b = GradientRecorder.apply(b, 'layer2')
		# 	c = GradientRecorder.apply(c, 'layer3')
		# 	d = GradientRecorder.apply(d, 'layer4')
		# 	output = GradientRecorder.apply(output, 'output')

		# 	self.logger.log_gradient_stats(self.grad_stats, self.forward_count)

		return F.tanh(output) * self.max_action

		# # Simple Version
		# a = self.relu(self.l1(state))
		# b = torch.tanh(self.l2(a))
		# return b * self.max_action

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		
		# # old version
		# # Q1 architecture
		# # self.l0 = nn.Linear(state_dim + action_dim, 1)
		# self.l1 = nn.Linear(state_dim + action_dim, 256)
		# self.l2 = nn.Linear(256, 128)
		# self.l3 = nn.Linear(128, 1)

		# # Q2 architecture
		# # self.l0_2 = nn.Linear(state_dim + action_dim, 1)
		# self.l4 = nn.Linear(state_dim + action_dim, 256)
		# self.l5 = nn.Linear(256, 128)
		# self.l6 = nn.Linear(128, 1)

		# self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

		# new version
		        
		# Q1
		self.l1 = nn.Linear(state_dim + action_dim, 128)  # 65 -> 128
		self.l2 = nn.Linear(128, 64)
		self.l3 = nn.Linear(64, 32)
		self.l4 = nn.Linear(32, 1)
		# self.l4 = nn.Linear(32, 16)
		# self.l5 = nn.Linear(16, 1)


		
		# Q2
		self.l6 = nn.Linear(state_dim + action_dim, 128)  # 65 -> 128
		self.l7 = nn.Linear(128, 64)
		self.l8 = nn.Linear(64, 32)
		self.l9 = nn.Linear(32, 1)
		# self.l9 = nn.Linear(32, 16)
		# self.l10 = nn.Linear(16, 1)


		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

		# # Simple Version
		# # Q1 
		# self.l1 = nn.Linear(state_dim + action_dim, 16)
		# self.l2 = nn.Linear(16, 1)

		# self.l3 = nn.Linear(state_dim + action_dim, 16)
		# self.l4 = nn.Linear(16, 1)
		
		self.relu = nn.ReLU()

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			if m in [self.l4, self.l9]:  # Q值输出层
				nn.init.uniform_(m.weight, -0.01, 0.01)
				nn.init.uniform_(m.bias, -0.01, 0.01)
			else:  # 隐藏层
				# 保持He初始化
				# nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
				nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
				if m.bias is not None:
					# 使用小的正值初始化偏置
					nn.init.constant_(m.bias, 0.1)

			# # 对Q1和Q2使用稍微不同的初始化以增加多样性
			# if m in [self.l4, self.l5, self.l6]:  # Q2的层
			# 	m.weight.data *= 1.1
			# 	if m.bias is not None:
			# 		m.bias.data *= 1.1
		
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)


	def forward(self, state, action):
		# # old version
		# sa = torch.cat([state, action], 1)

		# q1_1 = self.leaky_relu(self.l1(sa))
		# q1_2 = self.leaky_relu(self.l2(q1_1))
		# q1_3 = self.l3(q1_2)

		# q2_1 = self.leaky_relu(self.l4(sa))
		# q2_2 = self.leaky_relu(self.l5(q2_1))
		# q2_3 = self.l6(q2_2)
		
		# return q1_3, q2_3

		# new version
		if SKIP_GNN:
			state = state.reshape(-1, NUM_AREA * NUM_FEATURES).to(device)

		sa = torch.cat([state, action], 1)

		# Q1
		q1 = self.leaky_relu(self.l1(sa))
		# q1 = self.relu(self.l1(sa))
		# q1 = self.ln1(q1)

		q1 = self.leaky_relu(self.l2(q1))
		# q1 = self.relu(self.l2(q1))
		# q1 = self.ln2(q1)

		q1 = self.leaky_relu(self.l3(q1))
		# q1 = self.relu(self.l3(q1))
		# q1 = self.ln3(q1)

		q1 = self.l4(q1)
		# q1 = self.leaky_relu(self.l4(q1))
		# q1 = self.l5(q1)

		# Q2
		q2 = self.leaky_relu(self.l6(sa))
		# q2 = self.relu(self.l5(sa))
		# q2 = self.ln5(q2)

		q2 = self.leaky_relu(self.l7(q2))
		# q2 = self.relu(self.l6(q2))
		# q2 = self.ln6(q2)

		q2 = self.leaky_relu(self.l8(q2))
		# q2 = self.relu(self.l7(q2))
		# q2 = self.ln7(q2)

		q2 = self.l9(q2)
		# q2 = self.leaky_relu(self.l9(q2))
		# q2 = self.l10(q2)

		return q1, q2


	def Q1(self, state, action):
		# # old version
		# sa = torch.cat([state, action], 1)
		# q1_1 = self.leaky_relu(self.l1(sa))
		# q1_2 = self.leaky_relu(self.l2(q1_1))
		# q1_3 = self.l3(q1_2)
		# return q1_3

		# new version
		sa = torch.cat([state, action], 1)
		q1 = self.leaky_relu(self.l1(sa))
		# q1 = self.relu(self.l1(sa))
		# q1 = self.ln1(q1)

		q1 = self.leaky_relu(self.l2(q1))
		# q1 = self.relu(self.l2(q1))
		# q1 = self.ln2(q1) 

		q1 = self.leaky_relu(self.l3(q1))
		# q1 = self.relu(self.l3(q1))
		# q1 = self.ln3(q1)	

		q1 = self.l4(q1)
		# q1 = self.leaky_relu(self.l4(q1))
		# q1 = self.l5(q1)

		# # # Simple Version
		# sa = torch.cat([state, action], 1)
		# q1 = self.relu(self.l1(sa))
		# q1 = self.l2(q1)

		return q1
	
class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.001,
		policy_noise=0.3,
		noise_clip=0.5,
		policy_freq=2
	):
		if SKIP_GNN:
			gnn_output_dim = NUM_AREA * NUM_FEATURES
		else:
			gnn_output_dim = NUM_AREA * 8

		# self.gnn_auto_encoder = GNN_AutoEncoder(num_features=NUM_FEATURES, hidden_dim=128, output_dim=gnn_output_dim).to(device)
		# self.gnn_auto_encoder_optimizer = torch.optim.Adam(self.gnn_auto_encoder.parameters(), lr=gnn_lr)

		self.gnn_encoder = GNN_Encoder(num_features=NUM_FEATURES, hidden_dim=8, output_dim=gnn_output_dim).to(device)
		# self.gnn_encoder_optimizer = torch.optim.Adam(self.gnn_encoder.parameters(), lr=gnn_lr)

		# self.actor = Actor(state_dim=NUM_AREA*9, action_dim=1, max_action=0.5).to(device)
		self.actor = Actor(state_dim=gnn_output_dim, action_dim=action_dim, max_action=max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

		# self.critic = Critic(state_dim=NUM_AREA*9, action_dim=1).to(device)
		self.critic = Critic(state_dim=gnn_output_dim, action_dim=action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(list(self.critic.parameters()) + list(self.gnn_encoder.parameters()), lr=critic_lr)
		# self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
			
		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise * self.max_action
		self.noise_clip = noise_clip * self.max_action
		self.noise_decay = 0.995
		self.min_noise = 0.05

		self.policy_freq = policy_freq

		self.total_it = 0

		self.actor_loss = 0
		self.critic_loss = 0
		self.gnn_encoder_loss = 0

		self.mse_loss = nn.MSELoss()

		self.last_saved_iteration = 0

		# self.fixed_normalizer = FixedNormalizer_TRAIN()

		self.alpha = 0.8  # 重构损失权重
		self.beta = 0.5    # GNN representation loss weight
		self.gamma = 0.2  # 结构损失权重
		self.lambda_ = 1e-9  # 正则化损失权重
		self.action_penalty_coef = 0.00  # 动作惩罚系数
		# self.td3_weight = 2.0  # TD3损失权重
		# self.reward_loss_weight = 0.5  # 预测奖励损失权重

		self.recon_weight = 0.2
		self.struct_weight = 0.1
		self.ac_weight = 0.7

		self.training_logs = {
			'actor_loss': [],
			'critic_loss': [],
			'q1_mean': [],
			'q1_std': [],
			'q2_mean': [],
			'q2_std': [],
			'target_q_mean': [],
			'target_q_std': [],
			'reconstruction_loss': [],
			'structure_loss': [],
			'value_prediction_loss': [],
			'action_prediction_loss': [],
			'l2_reg_loss': []
		}

		# 添加学习率调度器
		self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			self.actor_optimizer, 
			mode='min',
			factor=0.5,    # 每次将学习率降低到原来的一半
			patience=5,     # 等待5个epoch无改善再降低学习率
			# verbose=True,
			min_lr=1e-6    # 最小学习率
		)

		self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			self.critic_optimizer,
			mode='min',
			factor=0.5,
			patience=5,
			# verbose=True,
			min_lr=1e-6
		)

		# self.gnn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		# 	# self.gnn_auto_encoder_optimizer,
		# 	self.gnn_encoder_optimizer,
		# 	mode='min',
		# 	factor=0.5,
		# 	patience=5,
		# 	# verbose=True,
		# 	min_lr=1e-6
		# )

		# 用于跟踪平均损失
		self.actor_losses = []
		self.critic_losses = []
		self.gnn_losses = []

	def select_action(self, state):
		state = state.to(device)
		if SKIP_GNN: 
			state = state.reshape(-1, NUM_AREA * NUM_FEATURES).to(device)
		action = self.actor(state)
		# print(f"real action:{action}")
		return action


	def train(self, replay_buffer, batch_size, edge_index_single):
		self.total_it += 1
			
		# get network
		edge_index_single = edge_index_single.to(device)

		# Sample replay buffer 
		state_ori, action, next_state_ori, reward, not_done, rej_rate, theta_value, weights, indices = replay_buffer.sample(batch_size)
		# move to device
		state_ori = state_ori.to(device) #[256,441]	
		next_state_ori = next_state_ori.to(device) #[256,441]
		reward = reward.to(device)
		action = action.to(device)
		not_done = not_done.to(device)
		rej_rate = rej_rate.to(device)
		theta_value = theta_value.to(device)

		# ---------------------- Critic Update ----------------------
		# 1. get critic states
		# with torch.no_grad():
		# 	edge_index = edge_index_single.to(device)
		# 	edge_index_batch_list = []
		# 	for i in range(batch_size):
		# 		area_offset = i * NUM_AREA
		# 		edge_index_i = edge_index + area_offset
		# 		edge_index_batch_list.append(edge_index_i)
		# 	edge_index_batch = torch.cat(edge_index_batch_list, dim=1) # [2, 10240] each graph contains 160 edges, 32 areas, 5 features

		# 	batch = torch.arange(batch_size,dtype=torch.long).unsqueeze(1).repeat(1, NUM_AREA).view(-1).to(device) #[2048] NUM_AREA*batch_size

		# 	state_critic, _, _ = self.gnn_auto_encoder(state_ori, edge_index_batch, batch_size, rej_rate, theta_value)
		# 	next_state_critic, _, _ = self.gnn_auto_encoder(next_state_ori, edge_index_batch, batch_size, rej_rate, theta_value)
		with torch.no_grad():
			state_critic = self.gnn_encoder(state_ori, edge_index_single, batch_size, rej_rate, theta_value)
			next_state_critic = self.gnn_encoder(next_state_ori, edge_index_single, batch_size, rej_rate, theta_value)

		# 2. calculate target Q
		with torch.no_grad():
			noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip).to(device)
			next_action = (self.actor_target(next_state_critic) + noise).clamp(-self.max_action, self.max_action)
			target_Q1, target_Q2 = self.critic_target(next_state_critic, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q
		
		# 3. calculate current Q
		current_Q1, current_Q2 = self.critic(state_critic, action)
		# critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		td_errors = torch.abs(current_Q1 - target_Q).detach().cpu().numpy()
		q_stats = {
			'q1_mean': current_Q1.mean().item(),
			'q1_std': current_Q1.std().item(),
			'q2_mean': current_Q2.mean().item(),
			'q2_std': current_Q2.std().item(),
			'target_q_mean': target_Q.mean().item(),
			'target_q_std': target_Q.std().item()
		}
		# print("Q value stats:", q_stats)
		# update priorities
		replay_buffer.update_priorities(indices, td_errors)
		critic_loss = (weights * F.mse_loss(current_Q1, target_Q, reduction='none')).mean() + \
				(weights * F.mse_loss(current_Q2, target_Q, reduction='none')).mean()

		self.critic_losses.append(critic_loss.item())

		# 4. optimize critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward(retain_graph=False)
		torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
		self.critic_optimizer.step()
		self.critic_loss = critic_loss.item()

		# 记录critic loss
		# self.training_logs['iterations'].append(self.total_it)
		self.training_logs['critic_loss'].append(critic_loss.item())
		self.training_logs['q1_mean'].append(q_stats['q1_mean'])
		self.training_logs['q1_std'].append(q_stats['q1_std'])
		self.training_logs['q2_mean'].append(q_stats['q2_mean'])
		self.training_logs['q2_std'].append(q_stats['q2_std'])
		self.training_logs['target_q_mean'].append(q_stats['target_q_mean'])
		self.training_logs['target_q_std'].append(q_stats['target_q_std'])

		# ---------------------- Actor and GNN AutoEncoder Update ----------------------
		if self.total_it % self.policy_freq == 0:
			# ---------------------- Actor Update ----------------------
			# 1. get actor states
			self.actor.grad_stats = {}
			# state_actor, _, _ = self.gnn_auto_encoder(state_ori, edge_index_batch, batch_size, rej_rate, theta_value)
			state_actor = self.gnn_encoder(state_ori, edge_index_single, batch_size, rej_rate, theta_value)

			# 2. freeze critic parameters
			for param in self.critic.parameters():
				param.requires_grad = False
			
			# 3. calculate actor loss
			actions = self.actor(state_actor)
			actor_loss = -self.critic.Q1(state_actor, actions).mean()

			# check Q values
			q_values = self.critic.Q1(state_actor, actions)
			print(f"Q values stats - mean: {q_values.mean()}, std: {q_values.std()}")

			action_penalty = (actions.pow(2)).mean()
			actor_loss = actor_loss + self.action_penalty_coef * action_penalty

			# 3.1 log activation and gradient stats
			for name, tensor in self.actor.activation_stats.items():
				if tensor.grad is not None:
					self.actor.grad_stats[name] = tensor.grad.norm().item()
					self.actor.activation_stats[name] = {
						'mean': tensor.mean().item(),
						'std': tensor.std().item(),
						'max': tensor.max().item(),
						'min': tensor.min().item()
					}
			self.actor.logger.log_gradient_stats(self.actor.grad_stats, self.total_it)

			# 4. optimize actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward(retain_graph=False)

			# self.actor.logger.log_gradients()

			# # 打印梯度统计
			# print("Gradient norms:")
			# for name, grad_norm in self.actor.grad_stats.items():
			# 	print(f"{name}: {grad_norm}")
				
			# # 也打印参数梯度
			# print("\nParameter gradients:")
			# for name, param in self.actor.named_parameters():
			# 	if param.grad is not None:
			# 		print(f"{name} grad norm: {param.grad.norm().item()}")
			# 	else:
			# 		print(f"{name} has no gradient")

			# # 安全地获取梯度统计
			# gradient_stats = {}
			# for name, layer in [('layer1', self.actor.l1), ('layer2', self.actor.l2)]:  
			# 	if layer.weight.grad is not None:
			# 		gradient_stats[f'{name}_grad'] = layer.weight.grad.abs().mean().item()
			# 	else:
			# 		print(f"Warning: No gradient for {name}")
			# 		gradient_stats[f'{name}_grad'] = 0.0

			# print("Gradient stats:", gradient_stats)
			# 4.1 add gradient clipping
			torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

			self.actor_optimizer.step()
			self.actor_loss = actor_loss.item()
			self.actor_losses.append(self.actor_loss)

			# 5. unfreeze critic parameters
			for param in self.critic.parameters():
				param.requires_grad = True
			
			# 6. update noise
			self.policy_noise = max(self.min_noise, self.policy_noise * self.noise_decay)

			# ---------------------- GNN AutoEncoder Update ----------------------
			if not SKIP_GNN:

				# 1. get GNN AutoEncoder states
				# state, state_decoded, adj_hat = self.gnn_auto_encoder(state_ori, edge_index_batch, batch_size, rej_rate, theta_value)
				# next_state, next_state_decoded, _ = self.gnn_auto_encoder(next_state_ori, edge_index_batch, batch_size, rej_rate, theta_value)

				state = self.gnn_encoder(state_ori, edge_index_single, batch_size, rej_rate, theta_value)
				next_state = self.gnn_encoder(next_state_ori, edge_index_single, batch_size, rej_rate, theta_value)

				# 2. calculate reconstruction loss
				# recon_loss = F.mse_loss(state_decoded, state_ori.view(batch_size,NUM_AREA,NUM_FEATURES))

				# 3. calculate structure loss
				# original_adj_single = to_dense_adj(edge_index_single, max_num_nodes=NUM_AREA)[0] #[NUM_AREA,NUM_AREA]
				# original_adj = original_adj_single.unsqueeze(0).expand(batch_size, NUM_AREA, NUM_AREA) #[batch_size,NUM_AREA,NUM_AREA]
				# struct_loss = self.structure_loss(original_adj, adj_hat)

				# 4. Calculate Actor and Critic Loss again for GNN AutoEncoder
				# 4.1 Calculate Critic Loss
				current_Q1, current_Q2 = self.critic(state, action)
				with torch.no_grad():
					noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip).to(device)
					next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
					target_Q1, target_Q2 = self.critic_target(next_state, next_action)
					target_Q = torch.min(target_Q1, target_Q2)
					target_Q = reward + not_done * self.discount * target_Q
				critic_loss_gnn = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

				# 4.2 Calculate Actor Loss
				actor_loss_gnn = -self.critic.Q1(state, self.actor(state)).mean()

				# 5. Update GNN AutoEncoder
				# ae_loss = (self.recon_weight * recon_loss + self.struct_weight * struct_loss + self.ac_weight * (critic_loss_gnn + actor_loss_gnn))
				ae_loss = (0.4 * critic_loss_gnn + 0.6 * actor_loss_gnn)

				self.gnn_losses.append(ae_loss.item())

				# self.gnn_auto_encoder_optimizer.zero_grad()
				# self.gnn_encoder_optimizer.zero_grad()
				# ae_loss.backward()

				# self.gnn_auto_encoder_optimizer.step()

				# self.gnn_encoder_optimizer.step()
				# self.gnn_encoder_loss = ae_loss.item()

				# self.training_logs['reconstruction_loss'].append(recon_loss.item())
				# self.training_logs['structure_loss'].append(struct_loss.item())


				# # 1. get GNN AutoEncoder states
				# state, state_decoded, adj_hat = self.gnn_auto_encoder(state_ori, edge_index_batch, batch_size, rej_rate, theta_value)
				# next_state, next_state_decoded, _ = self.gnn_auto_encoder(next_state_ori, edge_index_batch, batch_size, rej_rate, theta_value)

				# # 2. calculate reconstruction loss
				# # recon_loss = self.mse_loss(state_decoded, state_ori.view(batch_size,NUM_AREA,NUM_FEATURES))
				# recon_loss = F.mse_loss(state_decoded, state_ori.view(batch_size,NUM_AREA,NUM_FEATURES))

				# # 3. calculate structure loss
				# original_adj_single = to_dense_adj(edge_index_single, max_num_nodes=NUM_AREA)[0] #[NUM_AREA,NUM_AREA]
				# original_adj = original_adj_single.unsqueeze(0).expand(batch_size, NUM_AREA, NUM_AREA) #[batch_size,NUM_AREA,NUM_AREA]
				# struct_loss = self.structure_loss(original_adj, adj_hat)

				# # 4. calculate l2 regularization
				# # l2_reg = sum(p.pow(2.0).sum() for p in self.gnn_auto_encoder.parameters())
				# l2_reg = 0

				# # 5. calculate GNN representation quality loss
				# with torch.no_grad():
				# 	# 5.1 计算目标Q值
				# 	noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip).to(device)
				# 	next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
				# 	target_Q1, target_Q2 = self.critic_target(next_state, next_action)
				# 	target_Q = torch.min(target_Q1, target_Q2)
				# 	target_Q = reward + not_done * self.discount * target_Q

				# # 5.2 值预测损失
				# current_Q1, current_Q2 = self.critic(state, action)
				# value_prediction_loss = F.mse_loss(current_Q1, target_Q.detach()) + F.mse_loss(current_Q2, target_Q.detach())

				# # 5.3 动作预测损失
				# predicted_actions = self.actor(state)
				# action_prediction_loss = F.mse_loss(predicted_actions, action)

				# # 5.4 Print losses for reference
				# print(f"Reconstruction Loss: {recon_loss.item()}")
				# print(f"Structure Loss: {struct_loss.item()}")
				# print(f"L2 Regularization Loss: {l2_reg}")
				# print(f"Value Prediction Loss: {value_prediction_loss.item()}")
				# print(f"Action Prediction Loss: {action_prediction_loss.item()}")

				# # 6. calculate GNN AutoEncoder loss
				# ae_loss = (self.alpha * recon_loss + self.gamma * struct_loss + self.lambda_ * l2_reg \
				# 			+ self.beta * (value_prediction_loss + action_prediction_loss))

				# # 7. optimize GNN AutoEncoder
				# self.gnn_auto_encoder_optimizer.zero_grad()
				# ae_loss.backward()
				# self.gnn_auto_encoder_optimizer.step()
				# self.gnn_encoder_loss = ae_loss.item()
				# self.training_logs['reconstruction_loss'].append(recon_loss.item())
				# self.training_logs['structure_loss'].append(struct_loss.item())
				# self.training_logs['value_prediction_loss'].append(value_prediction_loss.item())
				# self.training_logs['action_prediction_loss'].append(action_prediction_loss.item())
				# self.training_logs['l2_reg_loss'].append(l2_reg)


			# 记录actor和autoencoder相关的loss
			self.training_logs['actor_loss'].append(actor_loss.item())

			# ---------------------- Update Target Networks ----------------------
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		else: # actor and gnn autoencoder not updated
			# fill in empty values for actor and autoencoder losses
			self.training_logs['actor_loss'].append(None)
			self.training_logs['reconstruction_loss'].append(None)
			self.training_logs['structure_loss'].append(None)
			self.training_logs['value_prediction_loss'].append(None)
			self.training_logs['action_prediction_loss'].append(None)
			self.training_logs['l2_reg_loss'].append(None)

		actor_loss, critic_loss, gnn_loss = self.get_epoch_losses()
		self.update_schedulers(actor_loss, critic_loss, gnn_loss)

		# original code
		if False:
			edge_index_gnn = edge_index.clone().detach()
			
			state_gnn = state_ori.clone().detach().requires_grad_(True)
			state_gnn, _, _ = self.gnn_auto_encoder(state_gnn, edge_index, batch_size, rej_rate, theta_value)
			# with torch.no_grad():
			# 	state, state_decoded = self.gnn_auto_encoder(state_ori, edge_index, batch_size)
			# 	next_state, next_state_decoded = self.gnn_auto_encoder(next_state_ori, edge_index, batch_size)
			state, state_decoded, adj_hat = self.gnn_auto_encoder(state_ori, edge_index, batch_size, rej_rate, theta_value)
			next_state, next_state_decoded, _ = self.gnn_auto_encoder(next_state_ori, edge_index, batch_size, rej_rate, theta_value)

			# # copy state and next_state for critic to avoid in-place operation
			# state_critic = state.clone().detach().requires_grad_(True)
			# next_state_critic = next_state.clone().detach().requires_grad_(True)
			# state_actor = state.clone().detach().requires_grad_(True)
			# next_state_actor = next_state.clone().detach().requires_grad_(True)
			state_critic = state
			next_state_critic = next_state
			state_actor = state
			next_state_actor = next_state

			with torch.no_grad():
				# Select action according to policy and add clipped noise
				noise = (
					torch.randn_like(action) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip).to(device)
				
				next_action = (
					self.actor_target(next_state_actor) + noise
				).clamp(-self.max_action, self.max_action)

				# Compute the target Q value
				target_Q1, target_Q2 = self.critic_target(next_state_critic, next_action)
				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = reward + not_done * self.discount * target_Q
				# print("not_done: ", not_done)

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state_critic, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			# critic_loss_gnn = critic_loss.clone().detach().requires_grad_(True)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward(retain_graph=False)
			# old_weight_l1 = copy.deepcopy(self.critic.l1.weight.data)
			
			self.critic_optimizer.step()
			# new_weight_l1 = self.critic.l1.weight.data
			# print(f"Critic L1 weights change: {torch.sum(new_weight_l1 - old_weight_l1)}")
		
			self.critic_loss = critic_loss.item()

			# calcualte predicted reward loss
			# reward_prediction_loss = F.mse_loss(predicted_reward.squeeze(), reward)
			reward_prediction_loss = 0

			# Delayed policy updates
			if self.total_it % self.policy_freq == 0:
				with torch.autograd.set_detect_anomaly(False):
					for param in self.critic.parameters(): #freeze critic
						param.requires_grad = False
					actor_loss = -self.critic.Q1(state_actor, self.actor(state_actor)).mean()
					
					self.actor_optimizer.zero_grad()
					actor_loss.backward(retain_graph=False)
					self.actor_optimizer.step()
					self.actor_loss = actor_loss.item()

					for param in self.critic.parameters():
						param.requires_grad = True # unfreeze critic

					original_adj = to_dense_adj(edge_index_gnn, max_num_nodes=NUM_AREA)[0].expand(batch_size, -1, -1)
					# print("state_ori: ", state_ori.size())
					# print("state_decoded: ", state_decoded.size())
					recon_loss = self.mse_loss(state_decoded, state_ori.view(batch_size,NUM_AREA,NUM_FEATURES))
					struct_loss = self.structure_loss(original_adj, adj_hat)
					l2_reg = sum(p.pow(2.0).sum() for p in self.gnn_auto_encoder.parameters())

					ae_loss = (self.alpha * recon_loss + 
							# self.beta * feat_loss + 
							self.gamma * struct_loss +
							self.lambda_ * l2_reg +
							self.reward_loss_weight * reward_prediction_loss)
					
					print("recon_loss: ", recon_loss*self.alpha)
					print("struct_loss: ", struct_loss*self.gamma)
					print("reward_prediction_loss: ", reward_prediction_loss*self.reward_loss_weight)
					print("l2_reg: ", l2_reg*self.lambda_)
					print("ae_loss: ", ae_loss)

					gnn_encoder_loss = ae_loss

					self.gnn_auto_encoder_optimizer.zero_grad()
					gnn_encoder_loss.backward()
					self.gnn_auto_encoder_optimizer.step()

					self.gnn_encoder_loss = gnn_encoder_loss.item()

					# l1_new_weights = self.gnn_auto_encoder.encoder.conv1.lin.weight.data
					# print(f"GNN Encoder Conv1 weights change: {torch.sum(l1_new_weights - l1_old_weights)}")

					# with open('losses.txt', 'a') as f:
					# 	f.write(f"Reconstruction Loss: {recon_loss}, Feature Loss: {feat_loss}, Structure Loss: {struct_loss}\n")

				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def update_schedulers(self, epoch_actor_loss=None, epoch_critic_loss=None, epoch_gnn_loss=None):
		"""
		在每个epoch结束时调用此方法来更新学习率
		"""
		if epoch_critic_loss is not None:
			self.critic_scheduler.step(epoch_critic_loss)
			
		if epoch_actor_loss is not None:
			self.actor_scheduler.step(epoch_actor_loss)
			
		# if epoch_gnn_loss is not None and not SKIP_GNN:
		# 	self.gnn_scheduler.step(epoch_gnn_loss)
	
	def get_epoch_losses(self):
		"""
		计算并返回这个epoch的平均损失
		"""
		actor_loss = np.mean(self.actor_losses) if self.actor_losses else None
		critic_loss = np.mean(self.critic_losses) if self.critic_losses else None
		gnn_loss = np.mean(self.gnn_losses) if self.gnn_losses else None
		
		# 清空损失列表，准备下一个epoch
		self.actor_losses = []
		self.critic_losses = []
		self.gnn_losses = []
		
		return actor_loss, critic_loss, gnn_loss
	
	def get_loss(self):
		return float(self.actor_loss), float(self.critic_loss), float(self.gnn_encoder_loss)
	
	def feature_loss(self, original, reconstructed):
		# return self.mse_loss(original, reconstructed)
		return F.mse_loss(original, reconstructed)

	def structure_loss(self, original_adj, reconstructed_adj):
		# return F.binary_cross_entropy_with_logits(reconstructed_adj, original_adj)
		return F.binary_cross_entropy(reconstructed_adj, original_adj) # adj is already sigmoided

	def save_training_logs(self, filename):
		"""保存或更新训练日志"""
		import json
		import os
		
		# 确保目录存在
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		
		current_length = len(self.training_logs['critic_loss'])
		if not hasattr(self, 'last_saved_iteration'):
			self.last_saved_iteration = 0
		
		# 准备新数据
		new_data = {k: [float(v) if v is not None else None for v in vals[self.last_saved_iteration:current_length]]
				for k, vals in self.training_logs.items()}
		
		try:
			# 尝试读取现有文件
			if os.path.exists(filename):
				with open(filename, 'r') as f:
					existing_data = json.load(f)
				# 合并现有数据和新数据
				for k in self.training_logs.keys():
					existing_data[k].extend(new_data[k])
				save_data = existing_data
			else:
				save_data = new_data
			
			# 保存合并后的数据
			with open(filename, 'w') as f:
				json.dump(save_data, f)
			
			# 更新最后保存的迭代次数
			self.last_saved_iteration = current_length
			
		except Exception as e:
			print(f"Error saving training logs: {e}")

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.critic_target.state_dict(), filename + "_critic_target")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
		torch.save(self.actor_target.state_dict(), filename + "_actor_target")

		# torch.save(self.gnn_auto_encoder.state_dict(), filename + "_gnn_auto_encoder")
		# torch.save(self.gnn_auto_encoder_optimizer.state_dict(), filename + "_gnn_auto_encoder_optimizer")
			
		torch.save(self.gnn_encoder.state_dict(), filename + "_gnn_encoder")
		# torch.save(self.gnn_encoder_optimizer.state_dict(), filename + "_gnn_encoder_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic", weights_only=True))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", weights_only=True))
	# self.critic_target = copy.deepcopy(self.critic)
		self.critic_target.load_state_dict(torch.load(filename + "_critic_target", weights_only=True))

		self.actor.load_state_dict(torch.load(filename + "_actor", weights_only=True))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", weights_only=True))
		# self.actor_target = copy.deepcopy(self.actor)
		self.actor_target.load_state_dict(torch.load(filename + "_actor_target", weights_only=True))
			
		if MISSING_REWARD_PREDICTOR:
			# 加载GNN自编码器的状态字典
			gnn_state_dict = torch.load(filename + "_gnn_auto_encoder", map_location='cpu')
			
			# 获取当前模型的状态字典
			current_state_dict = self.gnn_auto_encoder.state_dict()
			
			# 过滤掉reward_predictor的参数
			filtered_state_dict = {k: v for k, v in gnn_state_dict.items() 
								if k in current_state_dict and 'reward_predictor' not in k}
			
			# 使用strict=False加载过滤后的参数
			self.gnn_auto_encoder.load_state_dict(filtered_state_dict, strict=False)
		else:
			# self.gnn_auto_encoder.load_state_dict(torch.load(filename + "_gnn_auto_encoder", map_location='cpu'))
			self.gnn_encoder.load_state_dict(torch.load(filename + "_gnn_encoder", map_location='cpu'))

		if ENV_MODE == 'TRAIN':
			# self.gnn_auto_encoder_optimizer.load_state_dict(torch.load(filename + "_gnn_auto_encoder_optimizer", map_location='cpu'))
			# self.gnn_encoder_optimizer.load_state_dict(torch.load(filename + "_gnn_encoder_optimizer", map_location='cpu'))
			pass
	
	
	def load_dict(self, state_dict):
		self.gnn_encoder.load_state_dict(state_dict['gnn_encoder'])
		# self.gnn_encoder_optimizer.load_state_dict(state_dict['gnn_encoder_optimizer'])
		# self.gnn_auto_encoder.load_state_dict(state_dict['gnn_auto_encoder'])
		# self.gnn_auto_encoder_optimizer.load_state_dict(state_dict['gnn_auto_encoder_optimizer'])

		self.critic.load_state_dict(state_dict['critic'])
		self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
		# self.critic_target = copy.deepcopy(self.critic)
		self.critic_target.load_state_dict(state_dict['critic_target'])
		# self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

		self.actor.load_state_dict(state_dict['actor'])
		self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
		# self.actor_target = copy.deepcopy(self.actor)
		self.actor_target.load_state_dict(state_dict['actor_target'])
		# self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
			
	def state_dict(self):
		
		return {
			'actor': self.actor.state_dict(),
			'actor_target': self.actor_target.state_dict(),
			'actor_optimizer': self.actor_optimizer.state_dict(),

			'critic': self.critic.state_dict(),
			'critic_target': self.critic_target.state_dict(),
			'critic_optimizer': self.critic_optimizer.state_dict(),

			'gnn_encoder': self.gnn_encoder.state_dict(),
			# 'gnn_encoder_optimizer': self.gnn_encoder_optimizer.state_dict(),
			# 'gnn_auto_encoder': self.gnn_auto_encoder.state_dict(),
			# 'gnn_auto_encoder_optimizer': self.gnn_auto_encoder_optimizer.state_dict()
		}
	
	def to(self, device):
		self.actor.to(device)
		self.actor_target.to(device)
		# self.actor_optimizer.to(device)

		self.critic.to(device)
		self.critic_target.to(device)
		# self.critic_optimizer.to(device)

		# self.gnn_auto_encoder.to(device)
		self.gnn_encoder.to(device)

		return self

	def update(self, policy_params):
		self.load_dict(policy_params)

class TD3Logger:
	def __init__(self, log_dir='logs'):
		# Create logs directory if it doesn't exist
		os.makedirs(log_dir, exist_ok=True)
		
		# Create timestamp for unique log file names
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		
		# Set up file handler for activation stats
		activation_log_path = os.path.join(log_dir, f'td3_activation_stats_{timestamp}.log')
		self.activation_logger = logging.getLogger('td3_activation')
		self.activation_logger.setLevel(logging.INFO)
		
		activation_handler = logging.FileHandler(activation_log_path)
		activation_handler.setLevel(logging.INFO)

		# 梯度日志
		gradient_log_path = os.path.join(log_dir, f'td3_gradient_stats_{timestamp}.log')
		self.gradient_logger = logging.getLogger('td3_gradient')
		self.gradient_logger.setLevel(logging.INFO)
		
		gradient_handler = logging.FileHandler(gradient_log_path)
		gradient_handler.setLevel(logging.INFO)

		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		activation_handler.setFormatter(formatter)
		gradient_handler.setFormatter(formatter)

		# Remove any existing handlers to avoid duplicate logs
		self.activation_logger.handlers = []
		self.gradient_logger.handlers = []
		self.activation_logger.addHandler(activation_handler)
		self.gradient_logger.addHandler(gradient_handler)
		
		# Keep track of iteration count
		self.iteration = 0
		
	def log_activation_stats(self, stats_dict, iteration=None):
		"""
		Log activation statistics for each layer
		
		Args:
			stats_dict: Dictionary containing layer statistics
			iteration: Current training iteration (optional)
		"""
		if iteration is not None:
			self.iteration = iteration
		
		log_message = f"\nIteration {self.iteration}:\n"
		
		for layer_name, layer_stats in stats_dict.items():
			log_message += f"{layer_name}:\n"
			for stat_name, stat_value in layer_stats.items():
				log_message += f"  {stat_name}: {stat_value:.6f}\n"
		
		self.activation_logger.info(log_message)
		self.iteration += 1
	
	def log_gradient_stats(self, grad_stats, iteration=None):
		if iteration is not None:
			self.iteration = iteration
			
		log_message = f"\nIteration {self.iteration}:\n"
		for layer_name, grad_norm in grad_stats.items():
			log_message += f"{layer_name} gradient norm: {grad_norm:.6f}\n"
			
		self.gradient_logger.info(log_message)


# class FixedNormalizer_TRAIN(object):
#     def __init__(self):
#         path = 'mean_std_data_100800.npz'
#         assert os.path.exists(path), "mean_std_data_100800.npz not found"
#         data = np.load(path)
#         self.mean = torch.tensor(data['mean'], dtype=torch.float32).to(device)
#         self.std = torch.tensor(data['std'], dtype=torch.float32).to(device)
#         self.std[self.std == 0] = 1.0 # avoid division by zero

#     def __call__(self, x):
#         self.mean = self.mean.flatten()
#         self.std = self.std.flatten()
#         return (x - self.mean) / (self.std)
