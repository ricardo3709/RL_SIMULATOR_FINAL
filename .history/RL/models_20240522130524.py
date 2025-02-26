import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.nn import GCNConv, GNNPool, global_mean_pool
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

class GNN_Encoder(nn.Module):
    def __init__(self, num_features=4, hidden_dim=64, output_dim=32):
        super(GNN_Encoder, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch_index=None):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        if batch_index is not None:
            x = global_mean_pool(x, batch_index) #Pooling
        return x
    
class DDPG_Agent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)  # Assuming state_dim is the output of GNN
        self.critic = Critic(state_dim, action_dim)

    def update_policy(self, state, action, reward, next_state):
        # DDPG update logic
        pass

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x)) * self.max_action
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
