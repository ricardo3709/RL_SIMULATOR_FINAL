import torch
import numpy as np
import pickle
import pandas as pd
from src.simulator.config import *

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

veh_super_div = float(FLEET_SIZE[0]/10)
req_super_div = 150.0

def feature_preparation(state):
    with open(PATH_NODES, 'rb') as p:
        nodes = pickle.load(p)
    area_ids = AREA_IDS

    [avaliable_veh_areas, current_working_veh_areas, future_working_veh_areas_in_5, future_working_veh_areas_in_10 , gen_counts_areas, rej_counts_areas, attraction_counts_areas] = state

    # perform aggregation on counts
    gen_counts_areas_list = [sum(gen_counts_areas[i]) for i in range(NUM_AREA)]

    avaliable_veh_areas = torch.tensor(avaliable_veh_areas,dtype=torch.float32, device=device)/veh_super_div
    current_working_veh_areas = torch.tensor(current_working_veh_areas,dtype=torch.float32, device=device)/veh_super_div
    future_working_veh_areas_in_5 = torch.tensor(future_working_veh_areas_in_5,dtype=torch.float32, device=device)/veh_super_div
    future_working_veh_areas_in_10 = torch.tensor(future_working_veh_areas_in_10,dtype=torch.float32, device=device)/veh_super_div
    request_gen_counts_areas_col = torch.tensor(gen_counts_areas,dtype=torch.float32, device=device)/req_super_div
    request_rej_counts_areas_col = torch.tensor(rej_counts_areas,dtype=torch.float32, device=device)/req_super_div
    request_attraction_counts_areas_col = torch.tensor(attraction_counts_areas,dtype=torch.float32, device=device)/req_super_div

    features_tensor = torch.stack(
        [avaliable_veh_areas, 
         current_working_veh_areas, 
         future_working_veh_areas_in_5, 
         future_working_veh_areas_in_10, 
         request_gen_counts_areas_col, 
         request_rej_counts_areas_col, 
         request_attraction_counts_areas_col],
        dim=0
    ).transpose(0, 1).float()  # [NUM_AREA, NUM_FEATURES]

    return features_tensor