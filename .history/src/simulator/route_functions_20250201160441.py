"""
route planning functions
"""
import copy
import pickle
import os.path as osp
import pandas as pd
from src.simulator.config import *

# BASEDIR = osp.dirname(osp.abspath(__file__))
with open(PATH_ALL_PATH_MATRIX, 'rb') as p:
    ALL_PATH_MATRIX = pickle.load(p)

with open(PATH_ALL_PATH_TIME_MATRIX, 'rb') as t:
    ALL_PATH_TIME_MATRIX = pickle.load(t)

with open(PATH_NODE_ADJ_MATRIX, 'rb') as node_adj:
    NODE_ADJ_MATRIX = pickle.load(node_adj)

with open(PATH_NODES, 'rb') as n:
    NODES = pickle.load(n)

print(f"[INFO] Route functions are ready. ")

def check_problem_node(number_of_nodes: int):
    wrong_nodes_pair = []
    for i in range (0,number_of_nodes):
        for j in range (0,number_of_nodes):
            route = get_route(i,j)
            if route == []:
                wrong_nodes_pair.append([i,j])
    return wrong_nodes_pair

def get_route(origin: int, destination: int):
        # if MAP_NAME == "SmallGrid":
        #     route = pickle.loads(pickle.dumps(ALL_PATH_TABLE[origin][destination][0])) #deepcopy. accerlate with pickle
        #     # route = copy.deepcopy(ALL_PATH_TABLE[origin][destination][0])
        # elif MAP_NAME == "Manhattan":
        #     route = get_route_by_matrix(origin, destination, ALL_PATH_TABLE) #No need to deepcopy
        #     # route = copy.deepcopy(get_route_by_matrix(origin, destination, ALL_PATH_TABLE))
        # else:
        #     raise ValueError("Invalid MAP_NAME")
        # return route
        route = get_route_by_matrix(origin, destination, ALL_PATH_MATRIX) #No need to deepcopy
        return route
    
def get_timeCost(origin: int, destination: int):
        # if MAP_NAME == "SmallGrid":
        #     time = pickle.loads(pickle.dumps(ALL_PATH_TABLE[origin][destination][1]))*1.5 #deepcopy. accerlate with pickle
        #     # time = copy.deepcopy(ALL_PATH_TABLE[origin][destination][1]) * 1.5 
        # elif MAP_NAME == "Manhattan":
        #     time = ALL_PATH_TIME_MATRIX[origin][destination] #deepcopy. accerlate with pickle
        #     # time = copy.deepcopy(ALL_PATH_TIME_MATRIX[origin][destination])
        # else:
        #     raise ValueError("Invalid MAP_NAME")
        # return time
        time = ALL_PATH_TIME_MATRIX[origin][destination] #deepcopy. accerlate with pickle
        return time


def get_route_by_matrix(Oid: int, Did: int, all_path_matrix):
    current_node = Oid
    route = [Oid]
    # time_cost = 0.0

    while current_node != Did:
        next_node = pickle.loads(pickle.dumps(int(all_path_matrix[current_node][Did]))) #deepcopy. accerlate with pickle
        # next_node = copy.deepcopy(int(all_path_matrix[current_node][Did]))

        # time_to_next_node = all_path_time_matrix[current_node][next_node]
        if MAP_NAME == "Utrecht":
            if next_node == current_node:
                return []
        route.append(next_node)         
        # time_cost += time_to_next_node
        current_node = next_node

    return route

def get_surrounding_nodes(target_node, num_layers):
    connected_nodes = set()
    connected_nodes.add(target_node)
    
    for layer in range(1, num_layers+1):
        new_nodes = set()
        for node in connected_nodes:
            new_nodes.update(NODE_ADJ_MATRIX[node][NODE_ADJ_MATRIX[node] == 1].index)
        connected_nodes.update(new_nodes)
    
    return connected_nodes

def map_node_to_area(node_id):
    zone_id = NODES[NODES['node_id'] == node_id]['zone_id'].values[0]
    return zone_id
    # return node_lookup_table[node_lookup_table['node_id'] == node_id]['zone_id'].values[0]