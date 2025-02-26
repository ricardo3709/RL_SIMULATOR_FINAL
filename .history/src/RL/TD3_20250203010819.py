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

actor_lr = 1e-4
critic_lr = 1e-3
# gnn_lr = 1e-3

class GNN_Encoder(nn.Module):
    def __init__(self, num_features=NUM_FEATURES, hidden_dim=64, output_dim=256):
        super(GNN_Encoder, self).__init__()
        
        # GNN layers
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim*2)
        self.conv3 = GCNConv(hidden_dim*2, hidden_dim*4)
        
        # Normalization layers
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim*2)
        self.ln3 = nn.LayerNorm(hidden_dim*4)
        
        # Residual projection layers
        self.res1 = nn.Linear(num_features, hidden_dim)
        self.res2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.res3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        
        # Final projection
        self.fc = nn.Linear(hidden_dim*4 + NUM_FEATURES, int(output_dim/NUM_AREA))
        self.out_ln = nn.LayerNorm(int(output_dim/NUM_AREA))
        
    def forward(self, state, edge_index, batch_size, rej_rate, theta_value):
        if SKIP_GNN:
            return state
            
        x = state.to(device)
        edge_index = edge_index.to(device)
        x = x.reshape(batch_size, NUM_AREA, NUM_FEATURES)
        origin_graph = x.clone()

        # First GCN Layer with residual
        identity1 = self.res1(x)
        out1 = self.conv1(x, edge_index)
        x = F.leaky_relu(self.ln1(out1 + identity1))

        # Second GCN Layer with residual 
        identity2 = self.res2(x)
        out2 = self.conv2(x, edge_index)
        x = F.leaky_relu(self.ln2(out2 + identity2))

        # Third GCN Layer with residual
        identity3 = self.res3(x)
        out3 = self.conv3(x, edge_index)
        x = F.leaky_relu(self.ln3(out3 + identity3))

        # Concatenate with original features
        x = torch.cat((x, origin_graph), dim=2)

        # Final projection with normalization
        x = self.fc(x)
        x = self.out_ln(x)
        x = x.view(batch_size, -1)
        
        return F.tanh(x)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        # Wider and deeper network
        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 384)
        self.l3 = nn.Linear(384, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, action_dim)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(384)
        self.ln3 = nn.LayerNorm(256)
        self.ln4 = nn.LayerNorm(128)
        
        # Residual projection layers
        self.res1 = nn.Linear(state_dim, 512)
        self.res2 = nn.Linear(512, 384)
        self.res3 = nn.Linear(384, 256)
        self.res4 = nn.Linear(256, 128)
        
        self.max_action = max_action
        
    def forward(self, state):
        # First layer with residual
        identity1 = self.res1(state)
        out1 = self.l1(state)
        x = F.leaky_relu(self.ln1(out1 + identity1))
        
        # Second layer with residual
        identity2 = self.res2(x)
        out2 = self.l2(x)
        x = F.leaky_relu(self.ln2(out2 + identity2))
        
        # Third layer with residual
        identity3 = self.res3(x)
        out3 = self.l3(x)
        x = F.leaky_relu(self.ln3(out3 + identity3))
        
        # Fourth layer with residual
        identity4 = self.res4(x)
        out4 = self.l4(x)
        x = F.leaky_relu(self.ln4(out4 + identity4))
        
        # Final layer
        x = self.l5(x)
        return torch.tanh(x) * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 Architecture with increased width
        self.l1 = nn.Linear(state_dim + action_dim, 512)
        self.l2 = nn.Linear(512, 384)
        self.l3 = nn.Linear(384, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, 1)
        
        # Layer normalization for Q1
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(384)
        self.ln3 = nn.LayerNorm(256)
        self.ln4 = nn.LayerNorm(128)
        
        # Q2 Architecture (similar but separate)
        self.l6 = nn.Linear(state_dim + action_dim, 512)
        self.l7 = nn.Linear(512, 384)
        self.l8 = nn.Linear(384, 256)
        self.l9 = nn.Linear(256, 128)
        self.l10 = nn.Linear(128, 1)
        
        # Layer normalization for Q2
        self.ln5 = nn.LayerNorm(512)
        self.ln6 = nn.LayerNorm(384)
        self.ln7 = nn.LayerNorm(256)
        self.ln8 = nn.LayerNorm(128)

        # Residual connections for Q1
        self.res1 = nn.Linear(state_dim + action_dim, 512)
        self.res2 = nn.Linear(512, 384)
        self.res3 = nn.Linear(384, 256)
        self.res4 = nn.Linear(256, 128)
        
        # Residual connections for Q2
        self.res5 = nn.Linear(state_dim + action_dim, 512)
        self.res6 = nn.Linear(512, 384)
        self.res7 = nn.Linear(384, 256)
        self.res8 = nn.Linear(256, 128)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        # Q1 forward pass with residuals
        identity1 = self.res1(sa)
        q1 = self.l1(sa)
        q1 = F.leaky_relu(self.ln1(q1 + identity1))
        
        identity2 = self.res2(q1)
        q1 = self.l2(q1)
        q1 = F.leaky_relu(self.ln2(q1 + identity2))
        
        identity3 = self.res3(q1)
        q1 = self.l3(q1)
        q1 = F.leaky_relu(self.ln3(q1 + identity3))
        
        identity4 = self.res4(q1)
        q1 = self.l4(q1)
        q1 = F.leaky_relu(self.ln4(q1 + identity4))
        
        q1 = self.l5(q1)
        
        # Q2 forward pass with residuals
        identity5 = self.res5(sa)
        q2 = self.l6(sa)
        q2 = F.leaky_relu(self.ln5(q2 + identity5))
        
        identity6 = self.res6(q2)
        q2 = self.l7(q2)
        q2 = F.leaky_relu(self.ln6(q2 + identity6))
        
        identity7 = self.res7(q2)
        q2 = self.l8(q2)
        q2 = F.leaky_relu(self.ln7(q2 + identity7))
        
        identity8 = self.res8(q2)
        q2 = self.l9(q2)
        q2 = F.leaky_relu(self.ln8(q2 + identity8))
        
        q2 = self.l10(q2)
        
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        identity1 = self.res1(sa)
        q1 = self.l1(sa)
        q1 = F.leaky_relu(self.ln1(q1 + identity1))
        
        identity2 = self.res2(q1)
        q1 = self.l2(q1)
        q1 = F.leaky_relu(self.ln2(q1 + identity2))
        
        identity3 = self.res3(q1)
        q1 = self.l3(q1)
        q1 = F.leaky_relu(self.ln3(q1 + identity3))
        
        identity4 = self.res4(q1)
        q1 = self.l4(q1)
        q1 = F.leaky_relu(self.ln4(q1 + identity4))
        
        q1 = self.l5(q1)
        return q1