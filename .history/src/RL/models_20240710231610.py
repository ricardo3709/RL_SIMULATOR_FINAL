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

# torch.autograd.set_detect_anomaly(True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
actor_learning_rate = 1e-4
critic_learning_rate = 1e-3

class GNN_Encoder(nn.Module):
    def __init__(self, num_features=NUM_FEATURES, hidden_dim=64, output_dim=32):
        super(GNN_Encoder, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.ln1 = nn.LayerNorm(output_dim+NUM_FEATURES)
        self.pool = global_mean_pool  # Global mean pooling, to aggregate node features into graph features
        self.normalizer = Normalizer(size=output_dim+NUM_FEATURES)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, GCNConv):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float().to(device)
        origin_graph = x.clone()
        edge_index = edge_index.to(device)

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.cat((x, origin_graph), dim=1)
       
        # Normalize output features
        # for i in range(x.size(0)):
        #     self.normalizer.observe(x[i, :])
        # x = torch.stack([self.normalizer.normalize(x[i, :]) for i in range(x.size(0))])

        x = self.ln1(x)
        return x
    
    
    def forward_ori(self, data, last_rej_rate, last_theta):
        x, edge_index = data.x, data.edge_index
        x = x.float().to(device)
        edge_index = edge_index.to(device)

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        pooled_x = self.pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        # cat last rej and last theta
        # Ensure rejection_rate is a tensor and on the correct device
        rejection_rate = torch.tensor([last_rej_rate], dtype=torch.float32, device=x.device)
        last_theta = torch.tensor([last_theta], dtype=torch.float32, device=x.device)
        # Expand rejection_rate to match batch size
        rejection_rate = rejection_rate.expand(pooled_x.size(0), -1)
        last_theta = last_theta.expand(pooled_x.size(0), -1)
        # Concatenate pooled_x and rejection_rate
        output = torch.cat((pooled_x, rejection_rate), dim=-1)
        output = torch.cat((output, last_theta), dim=-1)
        
        # Normalize output features
        for i in range(output.size(0)):
            self.normalizer.observe(output[i, :])
        output = torch.stack([self.normalizer.normalize(output[i, :]) for i in range(output.size(0))])

        return output
    
class DDPG_Agent(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(DDPG_Agent, self).__init__()
        # Actor and Critic Networks
        self.actor = Actor(state_dim, action_dim, max_action)  # state_dim + 1 to account for rejection rate
        self.critic = Critic(state_dim, action_dim) # state_dim is the output dimension from GNN without rejection rate
        self.normalizer = Normalizer(size=state_dim)
        self.gnn_encoder = GNN_Encoder(num_features=NUM_FEATURES, hidden_dim=64, output_dim=state_dim)

        # Optimizer with different learning rates
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(list(self.critic.parameters()) + list(self.gnn_encoder.parameters()), lr=critic_learning_rate)

        # Target networks
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic_target = Critic(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Discount factor, higher value means higher importance to future rewards
        self.discount = 0.4

        # tau is the soft update parameter, higher value means higher importance to new weights
        # self.tau = 0.001
        self.tau = 0.01

        self.total_steps = 0


    def select_action(self, state):
        self.actor.eval()  # Set the actor network to evaluation mode
        with torch.no_grad():
             # Ensure 'state' is a tensor and on the correct device
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)  # Convert to tensor if it's not already
            # state = torch.FloatTensor(state)
            state = state.to(device)
            action = self.actor.forward(state).cpu()
        
        self.actor.train()  # Set the actor network back to training mode
        
        noise = torch.randn_like(action)/3*0.1 # most noise should be in [-0.2, 0.2]
        print(f"original action: {float(action)}, actor noise: {float(noise)}")
        # noise = np.expand_dims(noise, 0)  # Make noise the same shape as action
        action += noise  # Add noise for exploration
        action = float(action)

        return action

    def update_policy(self, last_reward, current_state, last_state):

        if not isinstance(last_reward, torch.Tensor):
            last_reward = torch.FloatTensor([last_reward])
        last_reward = last_reward.to(device).detach()

        if not isinstance(current_state, torch.Tensor):
            current_state = torch.FloatTensor(current_state)
        current_state = current_state.to(device).detach()

        if not isinstance(last_state, torch.Tensor):
            last_state = torch.FloatTensor(last_state)
        last_state = last_state.to(device).detach()

        # rename for clarity
        next_state = current_state
        current_state = last_state
        current_reward = last_reward

        # next_state = last_state
        # current_reward = last_reward

        # Calculate target Q
        with torch.no_grad():
            next_action = self.actor_target.forward(next_state)
            target_Q= current_reward + self.discount * self.critic_target.forward(next_state, next_action)

        with torch.autograd.set_detect_anomaly(False):
            # calculate critic loss and backward 
            current_action = self.actor.forward(current_state)
            current_Q = self.critic.forward(current_state, current_action)
            
            critic_loss = F.mse_loss(current_Q, target_Q.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=False)
            self.critic_optimizer.step()
            

            # critic(state, action) = Q(s,a)
            # self.actor(cat_state_encoded.detach()) = a
            # with torch.no_grad():
            # actor_loss = -self.critic(current_state.detach(), self.actor(current_state.detach())).mean() 
            current_action = self.actor.forward(current_state)
            actor_loss = -self.critic.forward(current_state, self.actor.forward(current_state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

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
        self.hid_size = 256
        self.hid_1_size = 128
        self.layer1 = nn.Linear(state_dim+NUM_FEATURES, self.hid_size)
        self.ln1 = nn.LayerNorm(self.hid_size)
        self.layer2 = nn.Linear(self.hid_size, self.hid_1_size)
        self.ln2 = nn.LayerNorm(self.hid_1_size)
        self.layer3 = nn.Linear(self.hid_1_size, action_dim)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, bias=False)
        
        self.max_action = max_action
        self.apply(self._init_weights)
        # self.normalizer = Normalizer(size=state_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m == self.layer3:
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
            else:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def forward(self, actor_x):
        actor_x = actor_x.to(device)

        # actor_x_normalized = actor_x.detach().clone().requires_grad_()
        # # Normalize input features
        # for i in range(actor_x_normalized.size(0)):
        #     self.normalizer.observe(actor_x_normalized[i, :])
        # actor_x = torch.stack([self.normalizer.normalize(actor_x_normalized[i, :]) for i in range(actor_x_normalized.size(0))])

        actor_x = torch.relu(self.ln1(self.layer1(actor_x)))
        actor_x = torch.relu(self.ln2(self.layer2(actor_x)))
        actor_x = torch.tanh(self.layer3(actor_x)) * self.max_action
        actor_x = actor_x.unsqueeze(0).permute(0,2,1) #[1,1,63]

        weights = self.conv1d(actor_x)
        # # Min-Max Normalize: [-1, 1]
        # min_val, _ = weights.min(dim=2, keepdim=True)
        # max_val, _ = weights.max(dim=2, keepdim=True)
        # weights = 2 * (weights - min_val) / (max_val - min_val) - 1
        weights = F.softmax(weights, dim=2) #normalize weights, sum to 1

        weighted_sum = torch.sum(weights * actor_x, dim=2) * 0.5 # scale to [-0.5, 0.5]

        final_action = torch.clamp(weighted_sum, -0.5, 0.5)
        assert -0.5 < final_action < 0.5
        # print(f"real action: {final_action}")
        # actor_x = torch.relu(self.layer3(actor_x)) * self.max_action
        return final_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.hid_2_size = 256
        self.hid_3_size = 128

        self.layer1 = nn.Linear(state_dim + NUM_FEATURES + action_dim, self.hid_2_size)
        self.ln1 = nn.LayerNorm(self.hid_2_size)
        self.layer2 = nn.Linear(self.hid_2_size, self.hid_3_size)
        self.ln2 = nn.LayerNorm(self.hid_3_size)
        self.layer3 = nn.Linear(self.hid_3_size, 1)

        # self.normalizer = Normalizer(size=state_dim + action_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))  
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, critic_x, u):
        critic_x = critic_x.to(device)
        u_expanded = u.expand(critic_x.size(0), -1).to(device)
        critic_x = torch.cat([critic_x, u_expanded], 1)

        # # Normalize input features
        # for i in range(critic_x.size(0)):
        #     self.normalizer.observe(critic_x[i, :])
        # critic_x = torch.stack([self.normalizer.normalize(critic_x[i, :]) for i in range(critic_x.size(0))])

        critic_x = torch.relu(self.ln1(self.layer1(critic_x)))
        critic_x = torch.relu(self.ln2(self.layer2(critic_x)))
        critic_x = self.layer3(critic_x)
        return critic_x

class Normalizer:
    def __init__(self, size, eps=1e-2):
        self.size = size
        self.eps = eps
        self.n = torch.zeros(1, device=device)
        self.mean = torch.zeros(size, device=device)
        self.mean_diff = torch.zeros(size, device=device)
        self.var = torch.zeros(size, device=device)

    def observe(self, x):
        self.n = self.n + 1  # Create a new tensor
        last_mean = self.mean.clone()
        self.mean = self.mean + (x - self.mean) / self.n
        self.mean_diff = self.mean_diff + (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clamp(min=self.eps)

    def normalize(self, x):
        obs_std = torch.sqrt(self.var)
        return (x - self.mean) / obs_std
    