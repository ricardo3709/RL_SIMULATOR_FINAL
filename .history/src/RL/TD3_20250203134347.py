import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
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
critic_lr = 3e-4
# gnn_lr = 1e-3

class GNN_Encoder(nn.Module):
    def __init__(self, num_features=NUM_FEATURES, hidden_dim=64, output_dim=64):
        super(GNN_Encoder, self).__init__()
        
        # GNN layers
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Normalization layers
        self.ln1 = nn.LayerNorm(hidden_dim)
        # self.ln2 = nn.LayerNorm(hidden_dim*2)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        # # Residual projection layers
        # self.res1 = nn.Linear(num_features, hidden_dim)
        # self.res2 = nn.Linear(hidden_dim, hidden_dim*2)
        # self.res3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        
        # Final projection
        self.fc = nn.Linear(hidden_dim, int(output_dim))
        # self.out_ln = nn.LayerNorm(int(output_dim/NUM_AREA))
        
    def forward(self, state, edge_index, batch_size, rej_rate, theta_value):
        if SKIP_GNN:
            return state
            
        x = state.to(device)
        edge_index = edge_index.to(device)
        x = x.reshape(batch_size, NUM_AREA, NUM_FEATURES)
        # origin_graph = x.clone()

        # First GCN Layer with no residual
        h1 = F.leaky_relu(self.ln1(self.conv1(x,edge_index)))

        # Second GCN Layer with residual connection
        h2 = F.leaky_relu(self.conv2(h1, edge_index)) + h1

        # cat h2 with original graph
        # h2 = torch.cat((x, origin_graph), dim=2)

        # Third GCN Layer with residual connection
        h3 = F.leaky_relu(self.ln3(self.conv3(h2, edge_index))) + h1

        # Final projection with normalization
        x = self.fc(h3)
        # x = self.out_ln(x)
        x = x.view(batch_size, -1)
        
        return F.tanh(x)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        # Wider and deeper network
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, action_dim)
        
        # Layer normalization
        # self.ln1 = nn.LayerNorm(512)
        # self.ln2 = nn.LayerNorm(384)
        # self.ln3 = nn.LayerNorm(256)
        # self.ln4 = nn.LayerNorm(128)
        
        # Residual projection layers
        # self.res1 = nn.Linear(state_dim, 512)
        # self.res2 = nn.Linear(512, 384)
        # self.res3 = nn.Linear(384, 256)
        # self.res4 = nn.Linear(256, 128)
        
        self.max_action = max_action
        
    def forward(self, state):
        # # First layer with residual
        # identity1 = self.res1(state)
        # out1 = self.l1(state)
        # x = F.leaky_relu(self.ln1(out1 + identity1))
        
        # # Second layer with residual
        # identity2 = self.res2(x)
        # out2 = self.l2(x)
        # x = F.leaky_relu(self.ln2(out2 + identity2))
        
        # # Third layer with residual
        # identity3 = self.res3(x)
        # out3 = self.l3(x)
        # x = F.leaky_relu(self.ln3(out3 + identity3))
        
        # # Fourth layer with residual
        # identity4 = self.res4(x)
        # out4 = self.l4(x)
        # x = F.leaky_relu(self.ln4(out4 + identity4))
        
        # # Final layer
        # x = self.l5(x)

        x = F.leaky_relu(self.l1(state))
        x = F.leaky_relu(self.l2(x))
        x = F.leaky_relu(self.l3(x))
        x = F.leaky_relu(self.l4(x))
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

        self.gnn_encoder = GNN_Encoder(num_features=NUM_FEATURES, hidden_dim=16, output_dim=gnn_output_dim).to(device)
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

        self.logger = TrainingLogger()

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

        critic_loss = (weights * F.mse_loss(current_Q1, target_Q, reduction='none')).mean() + \
                (weights * F.mse_loss(current_Q2, target_Q, reduction='none')).mean()
        
        self.logger.log_step(critic_loss=critic_loss.item())

        self.critic_losses.append(critic_loss.item())

        # 4. optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.gnn_encoder.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        self.critic_loss = critic_loss.item()

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
            self.logger.log_step(actor_loss=actor_loss.item())

            # check Q values
            q_values = self.critic.Q1(state_actor, actions)
            print(f"Q values stats - mean: {q_values.mean()}, std: {q_values.std()}")

            action_penalty = (actions.pow(2)).mean()
            actor_loss = actor_loss + self.action_penalty_coef * action_penalty

            # 4. optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=False)

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

            self.actor_optimizer.step()
            self.actor_loss = actor_loss.item()
            self.actor_losses.append(self.actor_loss)

            # 5. unfreeze critic parameters
            for param in self.critic.parameters():
                param.requires_grad = True
            
            # 6. update noise
            self.policy_noise = max(self.min_noise, self.policy_noise * self.noise_decay)

        actor_loss, critic_loss, gnn_loss = self.get_epoch_losses()
        self.update_schedulers(actor_loss, critic_loss, gnn_loss)
        self.logger.save_logs()
        
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

class TrainingLogger:
    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化记录字典
        self.logs = {
            'actor_loss': [],
            'critic_loss': [],
            'gnn_loss': [],
            'total_reward': [],
            'avg_action': [],
            'q_values': {
                'mean': [],
                'std': [],
                'min': [],
                'max': []
            }
        }
        
    def log_step(self, actor_loss=None, critic_loss=None, gnn_loss=None):
        """记录每个训练步骤的loss"""
        if actor_loss is not None:
            self.logs['actor_loss'].append(float(actor_loss))
        if critic_loss is not None:
            self.logs['critic_loss'].append(float(critic_loss))
        if gnn_loss is not None:
            self.logs['gnn_loss'].append(float(gnn_loss))
    
    def log_eval(self, total_reward, avg_action):
        """记录评估结果"""
        self.logs['total_reward'].append(float(total_reward))
        self.logs['avg_action'].append(float(avg_action))
    
    def log_q_values(self, q_mean, q_std, q_min, q_max):
        """记录Q值统计信息"""
        self.logs['q_values']['mean'].append(float(q_mean))
        self.logs['q_values']['std'].append(float(q_std))
        self.logs['q_values']['min'].append(float(q_min))
        self.logs['q_values']['max'].append(float(q_max))
    
    def save_logs(self):
        """保存日志到文件"""
        log_path = os.path.join(self.log_dir, 'training_logs.json')
        
        # 尝试读取现有日志
        existing_logs = {}
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                try:
                    existing_logs = json.load(f)
                except json.JSONDecodeError:
                    print("Warning: Could not read existing log file")
        
        # 合并日志
        for key in self.logs:
            if key not in existing_logs:
                existing_logs[key] = self.logs[key]
            else:
                # 处理嵌套字典的情况
                if isinstance(self.logs[key], dict):
                    if key not in existing_logs:
                        existing_logs[key] = {}
                    for sub_key in self.logs[key]:
                        if sub_key not in existing_logs[key]:
                            existing_logs[key][sub_key] = self.logs[key][sub_key]
                        else:
                            existing_logs[key][sub_key].extend(self.logs[key][sub_key])
                # 处理普通列表的情况
                else:
                    existing_logs[key].extend(self.logs[key])
                
        # 保存合并后的日志
        with open(log_path, 'w') as f:
            json.dump(existing_logs, f)
            
    def plot_curves(self):
        """绘制训练曲线"""
        # 创建一个2x2的子图布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Loss曲线
        ax1.plot(self.logs['actor_loss'], label='Actor Loss', alpha=0.7)
        ax1.plot(self.logs['critic_loss'], label='Critic Loss', alpha=0.7)
        ax1.plot(self.logs['gnn_loss'], label='GNN Loss', alpha=0.7)
        ax1.set_title('Training Losses')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Reward曲线
        ax2.plot(self.logs['total_reward'], label='Total Reward', color='green')
        ax2.set_title('Evaluation Reward')
        ax2.set_xlabel('Evaluation Episodes')
        ax2.set_ylabel('Total Reward')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Action分布
        ax3.plot(self.logs['avg_action'], label='Average Action', color='orange')
        ax3.set_title('Average Action Values')
        ax3.set_xlabel('Evaluation Episodes')
        ax3.set_ylabel('Action Value')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Q值统计
        ax4.plot(self.logs['q_values']['mean'], label='Q Mean', color='blue')
        ax4.fill_between(range(len(self.logs['q_values']['mean'])),
                        np.array(self.logs['q_values']['mean']) - np.array(self.logs['q_values']['std']),
                        np.array(self.logs['q_values']['mean']) + np.array(self.logs['q_values']['std']),
                        alpha=0.2, color='blue')
        ax4.plot(self.logs['q_values']['min'], '--', label='Q Min', color='red', alpha=0.5)
        ax4.plot(self.logs['q_values']['max'], '--', label='Q Max', color='green', alpha=0.5)
        ax4.set_title('Q-Value Statistics')
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Q-Value')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.close()