import torch
from models import GNN_Encoder, Actor, Critic
from environment import ManhattanTrafficEnv
from torch_geometric.data import Data
import numpy as np

def feature_preparation(state):
    [idle_cars, rebalancing_cars, request_gen_counts, request_rej_counts] = state

    idle_cars = torch.tensor(idle_cars)  # [4091]
    rebalancing_cars = torch.tensor(rebalancing_cars)  # [4091]
    request_gen_counts = torch.tensor(request_gen_counts)  # [4091]
    request_rej_counts = torch.tensor(request_rej_counts)  # [4091]

    # adjacency_matrix = np.array(adjacency_matrix)
    # edges = np.nonzero(adjacency_matrix)
    # edge_index = torch.tensor(edges, dtype=torch.long)

    # combine all features
    features = torch.stack((idle_cars, rebalancing_cars, request_gen_counts, request_rej_counts), dim=1)

    return features