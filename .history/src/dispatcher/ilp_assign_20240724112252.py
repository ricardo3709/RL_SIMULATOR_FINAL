"""
compute an assignment plan from all possible matches
"""

from src.dispatcher.scheduling import *
import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple
def ilp_assignment(veh_trip_pairs: List[Tuple[Veh, List[Req], List[Tuple[int, int, int, float]], float, float]], #list[[Veh, list[Req], list[(int, int, int, float)], float, float]]
                #    considered_rids: List[int],
                   reqs: List[Req],
                   vehs: List[Veh],
                   num_of_rejected_req_for_areas_dict_movingAvg,
                   num_of_generate_req_for_areas_dict_movingAvg,
                   config: ConfigManager) -> List[int]:
    # REWARD_THETA = config.get("REWARD_THETA")
    reward_theta = self.
    # Create a new model
    model = gp.Model("ilp")
    model.setParam("LogToConsole", 0) #0: no log output, 1: log output

    vehicle_check = [] #check if vehicle is assigned, 1 for assigned, 0 for not assigned
    request_check = [] #check if request is assigned, 1 for assigned, 0 for not assigned
    veh_trip_pairs_check = [] #check if vehicle-trip pair is assigned, 1 for assigned, 0 for not assigned
    selected_veh_trip_pair_indices = []
    
    # Declare Model Variables
    for i in range(len(vehs)):
        vehicle_check.append(0)
    for i in range(len(reqs)):
        request_check.append(0)
    for i in range(len(veh_trip_pairs)):
        veh_trip_pairs_check.append(model.addVar(vtype=GRB.CONTINUOUS))
    
    # Constrain 1: each vehicle can only be assigned at most one veh_trip_pair(request).
    for i in range(len(vehs)): #iterates over vehicles
        for j in range(len(veh_trip_pairs)): #iterates over vehicle-trip pairs
            if vehs[i] in veh_trip_pairs[j]: #veh_trip_pairs contain current vehicle
                vehicle_check[i] += veh_trip_pairs_check[j] 
        model.addConstr(vehicle_check[i] <= 1.0) #Number of constrain equal to number of vehicles
    
    # Constrain 2: each request can only be assigned to at most one veh_trip_pair(vehicle).
    for i in range(len(reqs)): #iterates over requests
        for j in range(len(veh_trip_pairs)): #iterates over vehicle-trip pairs
            if reqs[i] in veh_trip_pairs[j]: #veh_trip_pairs contain current request
                request_check[i] += veh_trip_pairs_check[j]
        model.addConstr(request_check[i] <= 1.0) #Number of constrain equal to number of requests
    
    # Objective: minimize the total delay of all veh_trip_pairs.
    object_score = 0.0

    for i in range(len(veh_trip_pairs)):\
        # Anticipatory Method Objective 
        cost = anticipatory_cost(veh_trip_pairs[i][2], veh_trip_pairs[i][0]) 
        reward = reward_function(veh_trip_pairs[i][2], veh_trip_pairs[i][0], num_of_rejected_req_for_areas_dict_movingAvg, num_of_generate_req_for_areas_dict_movingAvg, config)
        anticipate_method_object_score = cost - REWARD_THETA * reward
        object_score += veh_trip_pairs_check[i] * anticipate_method_object_score

    for req_check in request_check:
        object_score += (1 - req_check) * PENALTY 

    model.setObjective(object_score, GRB.MINIMIZE) #set the objective function to be minimized
    
    # Optimize model.
    model.optimize()
    
    # Get the result.
    for veh_trip_pair_idx, veh_trip_pairs_status in enumerate(veh_trip_pairs_check): 
        if veh_trip_pairs_status.getAttr(GRB.Attr.X) == 1:
            selected_veh_trip_pair_indices.append(veh_trip_pair_idx) #index of selected vehicle-trip pairs, optimized by ILP

    return selected_veh_trip_pair_indices #index of selected vehicle-trip pairs, optimized by ILP

def backup_ilp_assignment(veh_trip_pairs: List[Tuple[Veh, List[Req], List[Tuple[int, int, int, float]], float, float]], #list[[Veh, list[Req], list[(int, int, int, float)], float, float]]
                #    considered_rids: List[int],
                   reqs: List[Req],
                   vehs: List[Veh],
                   num_of_rejected_req_for_areas_dict_movingAvg,
                   num_of_generate_req_for_nodes_dict_movingAvg,
                   config: ConfigManager) -> List[int]:
    REWARD_THETA = config.get("REWARD_THETA")
    # print(f"using theta as: {REWARD_THETA}")
    # Create a new model
    # model = gp.Model("ilp")
    model = gp.Model("lp")
    model.setParam("LogToConsole", 0) #0: no log output, 1: log output

    vehicle_check = [] #check if vehicle is assigned, 1 for assigned, 0 for not assigned
    request_check = [] #check if request is assigned, 1 for assigned, 0 for not assigned
    veh_trip_pairs_check = [] #check if vehicle-trip pair is assigned, 1 for assigned, 0 for not assigned
    selected_veh_trip_pair_indices = []
    
    # Declare Model Variables
    for i in range(len(vehs)):
        # vehicle_check.append(model.addVar(vtype=GRB.BINARY)) # (lb = 0.0, ub = 1.0, obj = 0.0, vtype = GRB.BINARY)
        vehicle_check.append(model.addVar(vtype=GRB.CONTINUOUS)) # (lb = 0.0, ub = 1.0, obj = 0.0, vtype = GRB.BINARY)
    for i in range(len(reqs)):
        # request_check.append(model.addVar(vtype=GRB.BINARY))
        request_check.append(model.addVar(vtype=GRB.CONTINUOUS))
    for i in range(len(veh_trip_pairs)):
        # veh_trip_pairs_check.append(model.addVar(vtype=GRB.BINARY))
        veh_trip_pairs_check.append(model.addVar(vtype=GRB.CONTINUOUS))
    
    # Constrain 1: each vehicle can only be assigned at most one veh_trip_pair(request).
    for i in range(len(vehs)): #iterates over vehicles
        for j in range(len(veh_trip_pairs)): #iterates over vehicle-trip pairs
            # if veh_trip_pairs[j][1] == None: #empty assign
            #     continue
            if vehs[i] == veh_trip_pairs[j][0]: #veh_trip_pairs contain current vehicle
                vehicle_check[i] += veh_trip_pairs_check[j] 
        model.addConstr(vehicle_check[i] <= 1.0) #Number of constrain equal to number of vehicles
    
    # Constrain 2: each request can only be assigned to at most one veh_trip_pair(vehicle).
    for i in range(len(reqs)): #iterates over requests
        for j in range(len(veh_trip_pairs)): #iterates over vehicle-trip pairs
            # if veh_trip_pairs[j][1] == None: #empty assign
            #     continue
            if reqs[i] in veh_trip_pairs[j]: #veh_trip_pairs contain current request
                request_check[i] += veh_trip_pairs_check[j]
        model.addConstr(request_check[i] <= 1.0) #Number of constrain equal to number of requests
    
    

    # Objective: minimize the total delay of all veh_trip_pairs.
    object_score = 0.0

    for i in range(len(veh_trip_pairs)): #veh_trip_pairs = [veh, trip, sche, cost, score], len(veh_trip_pairs) = len(veh_trip_pairs_check)
        # if veh_trip_pairs[i][1] == None: #empty assign
        #     continue
        
        # Normal Method Objective
        # time_to_origin = get_timeCost(veh_trip_pairs[i][0].current_node, veh_trip_pairs[i][1].Ori_id) #time to travel to origin node, also need to be minimized
        # # object_score = Delay + Time_to_Origin(Pickup) + Penalty_of_Ignoring, for each veh_trip_pair
        # object_score += veh_trip_pairs_check[i] * (veh_trip_pairs[i][4] + time_to_origin) + (1.0 - veh_trip_pairs_check[i]) * PENALTY
        
        # Anticipatory Method Objective 
        
        cost = anticipatory_cost(veh_trip_pairs[i][2], veh_trip_pairs[i][0]) 
        reward = reward_function(veh_trip_pairs[i][2], veh_trip_pairs[i][0], num_of_rejected_req_for_areas_dict_movingAvg, num_of_generate_req_for_nodes_dict_movingAvg, config)
        anticipate_method_object_score = cost - float(REWARD_THETA) * reward
        object_score += veh_trip_pairs_check[i] * anticipate_method_object_score + (1.0 - veh_trip_pairs_check[i]) * PENALTY

    model.setObjective(object_score, GRB.MINIMIZE) #set the objective function to be minimized
    
    # Optimize model.
    model.optimize()

    # Get the result.
    for veh_trip_pair_idx, veh_trip_pairs_status in enumerate(veh_trip_pairs_check): 
        if veh_trip_pairs_status.getAttr(GRB.Attr.X) == 1:
            selected_veh_trip_pair_indices.append(veh_trip_pair_idx) #index of selected vehicle-trip pairs, optimized by ILP

    return selected_veh_trip_pair_indices #index of selected vehicle-trip pairs, optimized by ILP

    
    
    considered_rids = [req.Req_ID for req in reqs]
    if DEBUG_PRINT:
        print(f"                *ILP assignment with {len(veh_trip_pairs)} pairs...", end=" ")

    selected_veh_trip_pair_indices = []
    if len(veh_trip_pairs) == 0: #no vehicle-trip pair
        return selected_veh_trip_pair_indices
    
    # The order in which the vt-pairs are arranged can slightly affect the results. Reordering aims to eliminate it.
    veh_trip_pairs.sort(key=lambda e: e[0].id)

    try:
        # 1. Create a new model
        model = gp.Model("ilp")

        model.setParam("LogToConsole", 0) #0: no log output, 1: log output
        # model.setParam("Threads", 6)

        # 2. Create variables. vt_pair means vehicle-trip pair
        # A batch of requests is assigned to a vehicle-trip pair. We try to find the optimal assignment.
        var_vt_pair = []  # var_vt_pair[i] = 1 indicates selecting the i_th vehicle_trip_pair.
        for i in range(len(veh_trip_pairs)):
            var_vt_pair.append(model.addVar(vtype=GRB.BINARY)) # (lb = 0.0, ub = 1.0, obj = 0.0, vtype = GRB.BINARY)        
        var_order = []  # var_order[j] = 0 indicates assigning the i_th order in the list.
        for j in range(len(considered_rids)):
            var_order.append(model.addVar(vtype=GRB.BINARY)) # (lb = 0.0, ub = 1.0, obj = 0.0, vtype = GRB.BINARY)        

        # 3. Set objective: minimize Σ var_vt_pair[i] * score(vt_pair). var_vt_pair is binary, score is float.
        # Currently using delay as score, need to minimize delay.
        obj_expr = 0.0
        for i in range(len(veh_trip_pairs)):
            obj_expr += var_vt_pair[i] * veh_trip_pairs[i][4] # score
        # model.setObjective(obj_expr, GRB.MAXIMIZE) #set the objective function to be maximized
        model.setObjective(obj_expr, GRB.MINIMIZE) #set the objective function to be minimized

        # 4. Add constraints.
        #   (0) Get the pairs' indices for each vehicle and request
        vt_idx = [[] for v in range(len(vehs))]
        rt_idx = [[] for j in range(len(considered_rids))]
        for idx, [veh, trip, sche, cost, score] in enumerate(veh_trip_pairs):
            #skip all "empty assign" pairs
            if trip == None:
                continue
            vt_idx[veh.id].append(idx)
            req = trip #for Simonetto's Method, there is only one request in each trip. Save this line for future use.
                # DEBUG codes
                # if req.id not in considered_rids:
                #     print(f"[DEBUG2] req {req.Req_ID} {req.Status}, "
                #           f"request time {req.Req_Time}, latest pickup {req.Latest_PU_Time}, pickup time {req.Actual_PU_Time}")
                #     print(f"veh {veh.id}, at {veh.veh_time}s, onboard_rids {veh.onboard_rid}")
                #     print(f"sche {sche}")
                #     continue
            rt_idx[considered_rids.index(req.Req_ID)].append(idx)
        
        #   (1) Add constraint 1: each vehicle (v) can only be assigned at most one schedule (trip).
        #     Σ var_vt_pair[i] * Θ_vt(v) = 1, ∀ v ∈ V (Θ_vt(v) = 1 if v is in vt).
        #   iterates over vehicles. Each entry of vt_idx is a list of indices of vehicle-trip pairs.
        #   each vehicle may have multiple vehicle-trip pairs, but only one can be selected.
        #   thus, the sum of all selected vehicle-trip pairs for a vehicle should be less than or equal to 1.
        for indices in vt_idx: 
            con_this_veh = 0.0
            for i in indices:
                con_this_veh += var_vt_pair[i]
            model.addConstr(con_this_veh <= 1) #change to <= 1 to allow empty assign

        #   (2) Add constraint 2: each order/request (r) can only be assigned to at most one vehicle.
        #     Σ var_vt_pair[i] * Θ_vt(r) + var_order[j] = 1, ∀ r ∈ R. (Θ_vt(order) = 1 if r is in vt).
        for j, indices in enumerate(rt_idx):
            con_this_req = 0.0
            for i in indices:
                con_this_req += var_vt_pair[i]
            con_this_req += var_order[j]
            model.addConstr(con_this_req <= 1) #change to <= 1 to allow empty assign
            
        #   (3) Add constraint 3: no currently picking order is ignored.
        #     var_order[j] = 0, ∀ r ∈ R_picking
        if ensure_assigning_orders_that_are_picking:
            for j in range(len(considered_rids)):
                # if reqs[considered_rids[j]].status == OrderStatus.PICKING: #check status of all considered orders
                #     model.addConstr(var_order[j] == 0)
                if reqs[j].Status == OrderStatus.PICKING:
                    model.addConstr(var_order[j] == 0)

        # 5. Optimize model.
        model.optimize()

        # 6. Get the result.
        for i, var_vt in enumerate(var_vt_pair): #BUG: No. of vt_pairs with value 1 not equal to the number of reqs.
            if var_vt.getAttr(GRB.Attr.X) == 1:
                selected_veh_trip_pair_indices.append(i)

    except gp.GurobiError as e:
        print(f"\n[GUROBI] Error code = {str(e.message)} ({str(e)}).")
    except AttributeError:
        print("Encountered an attribute error")

    # if DEBUG_PRINT:
        # print(f"({timer_end(t)})")

    return selected_veh_trip_pair_indices #index of selected vehicle-trip pairs, optimized by ILP

def anticipatory_cost(new_schedule: list, veh: Veh):
    # new req
    PU_position = None
    DO_position = None
    for index, node in enumerate(new_schedule):
        if node[-1] == 'NEW_PU':
            PU_position = index
        elif node[-1] == 'NEW_DO':
            DO_position = index
            break
    # First term in objective function
    t_wait_new, detour_new = get_delay_and_detour_for_new_req(new_schedule, veh, PU_position, DO_position)
    first_term = PW * t_wait_new + PV * detour_new

    # For insertion in empty schedule
    if len(new_schedule) == 2: # only newly inserted req
        anticipatory_cost_value = first_term + PO * get_timeCost(new_schedule[0][0], new_schedule[1][0])
        return anticipatory_cost_value
    
    # Second term in objective function
    extra_delay_NEWPU = compute_extra_delay(new_schedule, PU_position, veh)
    extra_delay_NEWDO = compute_extra_delay(new_schedule, DO_position, veh)
    req_pair_dict = get_req_pair_dict(new_schedule)
    extra_delay, extra_detour = get_extra_delay_and_detour(new_schedule, veh, req_pair_dict, PU_position, DO_position, extra_delay_NEWPU, extra_delay_NEWDO)
    second_term = PW * extra_delay + PV * extra_detour

    # Third term in objective function
    third_term = PO * (extra_delay_NEWDO + extra_delay_NEWPU)
    
    # Anticipatory Cost Value
    anticipatory_cost_value = first_term + second_term + third_term
    return anticipatory_cost_value

def get_delay_and_detour_for_new_req(new_schedule: list, veh: Veh, PU_position: int, DO_position: int):
    time_to_first_node = get_timeCost(veh.current_node, new_schedule[0][0])
    new_req = new_schedule[PU_position]
    schedule_up_to_NEWPU = new_schedule[:PU_position+1]
    schedule_NEWPU_to_NEWDO = new_schedule[PU_position:DO_position+1]
    if PU_position == 0: # new req is the first node in the schedule
        t_wait_new = time_to_first_node
        detour_new = compute_schedule_time_cost(schedule_NEWPU_to_NEWDO) - new_req[4] #D(r,pi), new_req[4] is the shortest travel time
    else:     
        t_wait_new = compute_schedule_time_cost(schedule_up_to_NEWPU) #tw(r,pi)
        detour_new = compute_schedule_time_cost(schedule_NEWPU_to_NEWDO) - new_req[4] #D(r,pi), new_req[4] is the shortest travel time
    return t_wait_new, detour_new

def get_req_pair_dict(schedule: list):
    req_ids = (set(node[5] for node in schedule)) # get all request ids
    req_pair_dict = {req_id: [] for req_id in req_ids} # create a dictionary with request id as key

    for index in range(len(schedule)):
        node = schedule[index]
        req_id = node[5]
        req_type = node[1]
        req_pair_dict[req_id].append([index, req_type])
    req_pair_dict = dict(sorted(req_pair_dict.items(), key=lambda x: x[0])) # {req_id: [[index, 1(PU)], [index, -1(DO)]}

    return req_pair_dict

def get_extra_delay_and_detour(schedule: list, veh: Veh, req_pair_dict: dict, PU_position: int, DO_position: int, extra_delay_NEWPU: float, extra_delay_NEWDO: float):
    extra_delay= 0.0
    extra_detour = 0.0
    # for req_id, [PU_node, DO_node] in req_pair_dict.items():
    for req_id, nodes in req_pair_dict.items():
        for index, req_type in nodes:    
        # for index, req_type in [PU_node, DO_node]:
            if schedule[index][-1] == 'NEW_PU' or schedule[index][-1] == 'NEW_DO': #skip new req
                continue
            if index < PU_position+1: # before NEW_PU
                continue # no extra delay or detour

            else: # after NEW_PU
                if index < DO_position: # before NEW_DO
                    if req_type == 1: #pickup node
                        extra_delay += extra_delay_NEWPU
                    else: #dropoff node
                        if req_pair_dict[req_id][0][0] < PU_position: #PU of this req is before NEW_PU
                            extra_detour += extra_delay_NEWPU #extra detour for this DO node
                        else: #PU of this req is after NEW_PU
                            continue # no extra detour for this DO node, entire trip is in between NEW_PU and NEW_DO

                else: # after NEW_DO
                    if req_type == 1: #pickup node
                        extra_delay += extra_delay_NEWPU + extra_delay_NEWDO 
                    else: #dropoff node
                        if req_pair_dict[req_id][0][0] > DO_position: # PU of this req is after NEW_DO
                            continue # no extra detour for this DO node, entire trip is after finishing the new req
                        else: # PU of this req is before NEW_DO
                            extra_detour += extra_delay_NEWDO

    return extra_delay, extra_detour

def reward_function(new_schedule: list, veh: Veh, num_of_rejected_req_for_areas_dict_movingAvg, num_of_generate_req_for_areas_dict_movingAvg, config: ConfigManager):
    REWARD_TYPE = config.get("REWARD_TYPE")
    NODE_LAYERS = config.get("NODE_LAYERS")
    reward = 0.0
    #using Last Node for reward calculation
    last_node = new_schedule[-1]
    area_id = map_node_to_area(last_node[0])

    # scale reward to [0,1]
    total_reward = sum(sum(d) for d in num_of_rejected_req_for_areas_dict_movingAvg.values())
    reward = sum(num_of_rejected_req_for_areas_dict_movingAvg[area_id])
    if total_reward == 0:
        scaled_reward = 0.0
    else:
        scaled_reward = float(reward/total_reward)

    # Basic Reward Calculation
    # if REWARD_TYPE == 'REJ':
    #     reward = np.sum(num_of_rejected_req_for_areas_dict_movingAvg[area_id])
    # else: 
    #     reward = np.sum(num_of_generate_req_for_areas_dict_movingAvg[area_id])

    # Using Smooth Rejection Rate of Last Node as Reward
    # if REWARD_TYPE == 'REJ':
    #     reward = get_rejection_generation_rate_smooth(last_node[0], NODE_LAYERS, num_of_rejected_req_for_areas_dict_movingAvg)
    # else: 
    #     reward = get_rejection_generation_rate_smooth(last_node[0], NODE_LAYERS, num_of_generate_req_for_areas_dict_movingAvg)
    

    return scaled_reward

def get_rejection_generation_rate_smooth(target_node: int, num_layers: int, rate_dict: dict):
    considered_nodes = get_surrounding_nodes(target_node, num_layers)
    smooth_rate = 0.0
    for node in considered_nodes:
        smooth_rate += np.sum(rate_dict[node])/(PSI + get_timeCost(target_node, node))

    return smooth_rate


def greedy_assignment(veh_trip_pairs: List[Tuple[Veh, List[Req], List[Tuple[int, int, int, float]], float, float]]) -> List[int]:
    if DEBUG_PRINT:
        print(f"                *Greedy assignment with {len(veh_trip_pairs)} pairs...", end=" ")

    selected_veh_trip_pair_indices = []
    if len(veh_trip_pairs) == 0:
        return selected_veh_trip_pair_indices
    veh_trip_pairs.sort(key=lambda e: -e[4])

    selected_vids = []
    selected_rids = []

    for idx, [veh, trip, sche, cost, score] in enumerate(veh_trip_pairs):
        # Check if the vehicle has been selected.
        if veh.id in selected_vids:
            continue
        # Check if any request in the trip has been selected.
        T_id = [r.id for r in trip]
        if np.any([rid in selected_rids for rid in T_id]):
            continue
        # The current vehicle_trip_pair is selected.
        selected_vids.append(veh.id)
        selected_rids.extend(T_id)
        selected_veh_trip_pair_indices.append(idx)

    # if DEBUG_PRINT:
        # print(f"({timer_end(t)})")

    return selected_veh_trip_pair_indices
