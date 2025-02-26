import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GNNPool, global_mean_pool
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor_learning_rate = 1e-4
critic_learning_rate = 1e-3

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
        # Actor and Critic Networks
        self.actor = Actor(state_dim, action_dim)  # Assuming state_dim is the output of GNN
        self.critic = Critic(state_dim, action_dim)

        # Optimizer with different learning rates
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        # Target networks
        self.actor_target = Actor(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        # Discount factor, higher value means higher importance to future rewards
        self.discount = 0.99

        # tau is the soft update parameter, higher value means higher importance to new weights
        self.tau = 0.001

    def update_policy(self, state, action, reward, next_state):
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)

        # Calculate target Q
        with torch.no_grad():
            target_actions = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, target_actions)
            target_Q = reward + (self.discount * target_Q)

        # Update Critic
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


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
