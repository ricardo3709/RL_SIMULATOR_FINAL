"""
ILP Assignment for Rebalancer
"""

from src.dispatcher.scheduling import *
import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple



def rebalancer_ilp_assignment(veh_trip_pairs: List[Tuple[Veh, List[Req], List[Tuple[int, int, int, float]], float, float]], #list[[Veh, list[Req], list[(int, int, int, float)], float, float]]
                #    considered_rids: List[int],
                   reqs: List[Req],
                   vehs: List[Veh]) -> List[int]:
    
    # Create a new model
    # model = gp.Model("rebalancer_ilp")
    model = gp.Model("rebalancer_lp")
    model.setParam("LogToConsole", 0) #0: no log output, 1: log output

    vehicle_check = [] #check if vehicle is assigned, 1 for assigned, 0 for not assigned
    request_check = [] #check if request is assigned, 1 for assigned, 0 for not assigned
    veh_trip_pairs_check = [] #check if vehicle-trip pair is assigned, 1 for assigned, 0 for not assigned
    selected_veh_trip_pair_indices = []

    # Declare Model Variables
    for i in range(len(vehs)):
        vehicle_check.append(model.addVar(vtype=GRB.CONTINUOUS)) # (lb = 0.0, ub = 1.0, obj = 0.0, vtype = GRB.BINARY)
    for i in range(len(reqs)):
        request_check.append(model.addVar(vtype=GRB.CONTINUOUS))
    for i in range(len(veh_trip_pairs)):
        veh_trip_pairs_check.append(model.addVar(vtype=GRB.CONTINUOUS))
    
    # Constrain 1: each vehicle can only be assigned at most one veh_trip_pair(request).
    for i in range(len(vehs)): #iterates over vehicles
        for j in range(len(veh_trip_pairs)): #iterates over vehicle-trip pairs
            # if veh_trip_pairs[j][1] == None: #empty assign
            #     continue
            if vehs[i] == veh_trip_pairs[j][0]: #veh_trip_pairs contain current vehicle
                vehicle_check[i] += veh_trip_pairs_check[j] 
        model.addConstr(vehicle_check[i] <= 1.0) #Number of constrain equal to number of vehicles, each veh can only be assigned at most one veh_trip_pair
    
    # Constrain 2: each request can only be assigned to at most one veh_trip_pair(vehicle).
    for i in range(len(reqs)): #iterates over requests
        for j in range(len(veh_trip_pairs)): #iterates over vehicle-trip pairs
            # if veh_trip_pairs[j][1] == None: #empty assign
            #     continue
            if reqs[i] in veh_trip_pairs[j]: #veh_trip_pairs contain current request
                request_check[i] += veh_trip_pairs_check[j]
        model.addConstr(request_check[i] <= 1.0) #Number of constrain equal to number of requests, each req can only be assigned to at most one veh_trip_pair
    
    # Objective: minimize the total delay of all veh_trip_pairs.
    object_score = 0.0

    for i in range(len(veh_trip_pairs)): #veh_trip_pairs = [veh, trip, sche, cost, score], len(veh_trip_pairs) = len(veh_trip_pairs_check)
        # if veh_trip_pairs[i][1] == None: #empty assign
        #     continue
        # time_to_origin = retrive_TimeCost(veh_trip_pairs[i][0].current_node, veh_trip_pairs[i][1].Ori_id) #time to travel to origin node, also need to be minimized
        # object_score = Delay + Time_to_Origin(Pickup) + Penalty_of_Ignoring, for each veh_trip_pair
        time_to_origin = veh_trip_pairs[i][3]
        object_score += veh_trip_pairs_check[i] * (time_to_origin) + (1.0 - veh_trip_pairs_check[i]) * REBALANCER_PENALTY
    
    model.setObjective(object_score, GRB.MINIMIZE) #set the objective function to be minimized
    
    # Optimize model.
    model.optimize()

    # Get the result.
    for veh_trip_pair_idx, veh_trip_pairs_status in enumerate(veh_trip_pairs_check): 
        if veh_trip_pairs_status.getAttr(GRB.Attr.X) == 1:
            selected_veh_trip_pair_indices.append(veh_trip_pair_idx) #index of selected vehicle-trip pairs, optimized by ILP

    return selected_veh_trip_pair_indices #index of selected vehicle-trip pairs, optimized by ILP
