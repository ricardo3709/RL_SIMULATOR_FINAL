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

    request_gen_counts = [sum(d)for d in gen_counts_areas.values()]
    request_rej_counts = [sum(d)for d in rej_counts_areas.values()]
    request_attraction_counts = [sum(d)for d in attraction_counts_areas.values()]

    # avaliable_veh_areas = np.zeros((NUM_AREA,2), dtype=np.float32)
    # avaliable_veh_areas[:,0] = area_ids
    # working_veh_areas = np.zeros((NUM_AREA,2), dtype=np.float32)
    # working_veh_areas[:,0] = area_ids


    # for idx, counts in enumerate(avaliable_veh_nodes):
    #     node_id = idx + 1 
    #     area_id = nodes[nodes['node_id'] == node_id]['zone_id'].values[0]
    #     indice = np.where(avaliable_veh_areas[:,0] == area_id)[0]
    #     avaliable_veh_areas[indice,1] += counts
        
    
    # for idx, counts in enumerate(working_veh_nodes):
    #     node_id = idx + 1 
    #     area_id = nodes[nodes['node_id'] == node_id]['zone_id'].values[0]
    #     indice = np.where(working_veh_areas[:,0] == area_id)[0]
    #     working_veh_areas[indice,1] += counts
   
    # avaliable_veh_areas_col = torch.tensor(avaliable_veh_areas[:,1],dtype=torch.float32, device=device)/veh_super_div
    # working_veh_areas_col = torch.tensor(working_veh_areas[:,1],dtype=torch.float32, device=device)/veh_super_div
    # request_gen_counts_areas_col = torch.tensor(request_gen_counts,dtype=torch.float32, device=device)/req_super_div
    # request_rej_counts_areas_col = torch.tensor(request_rej_counts,dtype=torch.float32, device=device)/req_super_div
    # request_attraction_counts_areas_col = torch.tensor(request_attraction_counts,dtype=torch.float32, device=device)/req_super_div

    avaliable_veh_areas_col = torch.tensor(avaliable_veh_areas,dtype=torch.float32, device=device)/veh_super_div
    working_veh_areas_col = torch.tensor(current_working_veh_areas,dtype=torch.float32, device=device)/veh_super_div
    
    features_tensor = torch.stack(
        [avaliable_veh_areas_col,
        working_veh_areas_col,
        request_gen_counts_areas_col,
        request_rej_counts_areas_col,
        request_attraction_counts_areas_col],
        dim=0
    ).transpose(0, 1).float()  # [NUM_AREA, 5]

    return features_tensor