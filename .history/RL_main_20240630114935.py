import torch
from src.RL.models import GNN_Encoder,DDPG_Agent
from src.RL.environment import ManhattanTrafficEnv
from src.RL.train import train
from multiprocessing_simulator import multi_process_test
from torch_geometric.data import Data
import numpy as np
import os
from src.simulator.config import *

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    environment = ManhattanTrafficEnv()
    
    # Setup dimensions
    state_dim = 32  # Assume output dimension from GNN
    # action_dim = environment.action_space.shape[0]
    action_dim = 1
    max_action = float(environment.action_space.high[0])

    # Initialize models
    gnn_encoder = GNN_Encoder(num_features=NUM_FEATURES, hidden_dim=64, output_dim=state_dim).to(device)
    ddpg_agent = DDPG_Agent(state_dim=state_dim, action_dim=action_dim, max_action= max_action).to(device)
    # actor = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    # critic = Critic(state_dim=state_dim, action_dim=action_dim)

    models = (gnn_encoder,ddpg_agent)

    # Setup optimizer
    optimizer = torch.optim.Adam(
        list(gnn_encoder.parameters()) + list(ddpg_agent.actor.parameters()) + list(ddpg_agent.critic.parameters()), lr=1e-4
    )

    # # Train the model
    # if WARM_UP_EPOCHS > 0:
    #     warm_up_rejections = multi_process_test(models, environment, epochs = WARM_UP_EPOCHS)
    #     total_warm_up_rej = []
    #     for item in warm_up_rejections:
    #         for rej in item._value:
    #             total_warm_up_rej.append(rej)
    #     environment.past_rejections.extend(total_warm_up_rej)

    train(models, environment, epochs=500)
    # train(gnn_encoder, actor, critic, environment, epochs=100, optimizer=optimizer)
