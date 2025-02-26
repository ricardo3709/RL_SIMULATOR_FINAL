import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
from src.simulator.config import *

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor_learning_rate = 1e-4
critic_learning_rate = 1e-3

class GNN_Encoder(nn.Module):
    def __init__(self, num_features=NUM_FEATURES, hidden_dim=64, output_dim=32):
        super(GNN_Encoder, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.pool = global_mean_pool  # Global mean pooling, to aggregate node features into graph features

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float().to(device)
        edge_index = edge_index.to(device)
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        pooled_x = self.pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        
        return pooled_x
    
class DDPG_Agent(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(DDPG_Agent, self).__init__()
        # Actor and Critic Networks
        self.actor = Actor(state_dim, action_dim, max_action)  # Assuming state_dim is the output of GNN
        self.critic = Critic(state_dim, action_dim)

        self.gnn_encoder = GNN_Encoder(num_features=NUM_FEATURES, hidden_dim=64, output_dim=state_dim)

        # Optimizer with different learning rates
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        # Target networks
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic_target = Critic(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Discount factor, higher value means higher importance to future rewards
        self.discount = 0.99

        # tau is the soft update parameter, higher value means higher importance to new weights
        self.tau = 0.001

        self.total_steps = 0

    def select_action(self, state):
        self.actor.eval()  # Set the actor network to evaluation mode
        with torch.no_grad():
             # Ensure 'state' is a tensor and on the correct device
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)  # Convert to tensor if it's not already
            # state = torch.FloatTensor(state)
            state = state.to(device)
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()  # Set the actor network back to training mode
        return action

    def update_policy(self, state, action, reward, next_state, edge_index):
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor([reward]).to(device)
        
        graph_state = Data(x=state, edge_index=edge_index)
        graph_next_state = Data(x=next_state, edge_index=edge_index)
        state_encoded = self.gnn_encoder(graph_state)
        next_state_encoded = self.gnn_encoder(graph_next_state)

        # Calculate target Q
        with torch.no_grad():
            target_actions = self.actor_target(next_state_encoded)
            target_Q = self.critic_target(next_state_encoded, target_actions)
            target_Q = reward + (self.discount * target_Q)

        # 首先计算 Actor 的损失，但不立即进行反向传播
        actor_loss = -self.critic(state_encoded, self.actor(state_encoded)).mean()

        # 接着计算 Critic 的损失并进行反向传播
        current_Q = self.critic(state_encoded, action)
        critic_loss = pickle.loads(pickle.dumps(F.mse_loss(current_Q, target_Q))) # Deepcopy the loss to avoid in-place operations
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 然后对 Actor 进行反向传播
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # # Update Critic
        # current_Q = self.critic(state_encoded, action)
        # critic_loss = F.mse_loss(current_Q, target_Q)
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()

        # # Update Actor
        # actor_loss = -self.critic(state_encoded, self.actor(state_encoded)).mean()
        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()

        # Logging
        if self.total_steps % 100 == 0:
            print(f"Step {self.total_steps}: Critic Loss = {critic_loss.item()}, Actor Loss = {actor_loss.item()}")


        # Soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, actor_x):
        actor_x = torch.relu(self.layer1(actor_x))
        actor_x = torch.relu(self.layer2(actor_x))
        actor_x = torch.tanh(self.layer3(actor_x)) * self.max_action
        return actor_x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

    def forward(self, critic_x, u):
        critic_x = torch.cat([critic_x, u], 1)
        critic_x = torch.relu(self.layer1(critic_x))
        critic_x = torch.relu(self.layer2(critic_x))
        critic_x = self.layer3(critic_x)
        return critic_x
