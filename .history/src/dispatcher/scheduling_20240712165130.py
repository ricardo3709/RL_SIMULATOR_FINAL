"""
compute all feasible schedules for a given vehicle v and a trip T.
"""
import copy

from src.simulator.request import Req
from src.simulator.vehicle import Veh
from src.simulator.route_functions import *
from src.utility.utility_functions import *
from src.simulator.config import *
from typing import List, Tuple, Union

def compute_schedule(veh: Veh, req: Req): 
    # veh.schedule = veh.remove_duplicate_sublists(veh.schedule)
    # current_schedule = copy.deepcopy(veh.schedule) 
    if veh.status == VehicleStatus.REBALANCING: #temp clear rebalancing schedule, if not assigned, schedule will not be changed
        current_schedule = []
    else:
        current_schedule = pickle.loads(pickle.dumps(veh.schedule))
    # assert len(current_schedule) < 5 #DEBUG CODE

    if current_schedule == []: #if the vehicle has no schedule
        # For new req, one extra elements at last. Use in anticipatory ILP
        current_schedule.append([req.Ori_id, 1, req.Num_people, req.Latest_PU_Time, req.Shortest_TT, req.Req_ID, 'NEW_PU']) 
        current_schedule.append([req.Des_id, -1, req.Num_people, req.Latest_DO_Time, req.Shortest_TT, req.Req_ID, 'NEW_DO'])
        constrain_flag, time_cost = test_constraints(current_schedule, veh) #this time_cost includes time for veh travelling from current node to first node in new_schedule
        if constrain_flag:
            return current_schedule, time_cost
    
    #schedule is not empty, remove all 'NEW_PU' and 'NEW_DO' marks in nodes
    for index in range(len(current_schedule)):
        current_schedule[index] = current_schedule[index][:6]#remove the sixth element
        
    if HEURISTIC_ENABLE and len(current_schedule) > MAX_SCHEDULE_LENGTH: #if the vehicle has more than 30 nodes in the schedule
        return compute_schedule_by_half(veh, req, current_schedule)
    else:
        return compute_schedule_normal(veh, req, current_schedule)
    


def compute_schedule_normal(veh: Veh, req: Req, current_schedule: list):
    # feasible_schedules = []
    best_schedule = None
    min_time_cost = np.inf 

    for PU_node_position in range(len(current_schedule) + 1):
        for DO_node_position in range(PU_node_position + 1, len(current_schedule) + 2):
            new_schedule = insert_request_into_schedule(current_schedule, req, PU_node_position, DO_node_position)
            constrain_flag, time_cost = test_constraints(new_schedule, veh) #this time_cost includes time for veh travelling from current node to first node in new_schedule
            if constrain_flag: #check if the new schedule satisfies all constraints
                # feasible_schedules.append(new_schedule) #store the feasible schedule, but not the best one
                if time_cost < min_time_cost: #update the best schedule
                    # best_schedule = copy.deepcopy(new_schedule)
                    best_schedule = new_schedule #try shallow copy first. If it doesn't work, use deep copy
                    min_time_cost = time_cost

    return best_schedule, min_time_cost

def compute_schedule_by_half(veh: Veh, req: Req, current_schedule: list):
    feasible_schedules = []
    best_schedule = None
    best_PU_node_position = None
    flag_PU = False

    # Consider PU node pisition first, find best one
    for PU_node_position in range(len(current_schedule) + 1):
        new_schedule = insert_request_into_schedule_half(current_schedule, req, PU_node_position, None)
        # time_cost = compute_schedule_time_cost(new_schedule)
        extra_delay = compute_extra_delay(new_schedule, PU_node_position, veh)
        feasible_schedules.append([new_schedule, extra_delay, PU_node_position]) #store the feasible schedule, but not the best one
    
    if len(feasible_schedules) == 0: #if no feasible schedule found
        return None, None
    
    # Sort feasible schedules by extra_delay, minimize extra_delay
    feasible_schedules.sort(key = lambda x: x[1])

    for new_schedule, extra_delay, PU_node_position in feasible_schedules:
        constrain_flag, _ = test_constraints(new_schedule, veh)
        if constrain_flag: #check if the new schedule satisfies all constraints
           # found the best PU node position
           flag_PU = True
           best_PU_node_position = PU_node_position
           best_schedule = new_schedule
           break

    if not flag_PU: #if no feasible schedule found
        return None, None
    
    feasible_schedules = [] #clear the list
    # Consider DO node position, find best one
    for DO_node_position in range(best_PU_node_position + 1, len(current_schedule) + 2):
        new_schedule = insert_request_into_schedule_half(best_schedule, req, best_PU_node_position, DO_node_position)
        # time_cost = compute_schedule_time_cost(new_schedule)
        extra_delay = compute_extra_delay(new_schedule, DO_node_position, veh)
        feasible_schedules.append([new_schedule, extra_delay, DO_node_position]) #store the feasible schedule, but not the best one
    
    if len(feasible_schedules) == 0: #if no feasible schedule found
        return None, None
    
    # Sort feasible schedules by time cost, minimize time cost
    feasible_schedules.sort(key = lambda x: x[1])
    
    for new_schedule, extra_delay, PU_node_position in feasible_schedules:
        constrain_flag, time_cost = test_constraints(new_schedule, veh)
        if constrain_flag: #check if the new schedule satisfies all constraints
           # found the best DO node position
           return new_schedule, time_cost
           
    return None, None

def score_vt_pair_with_delay(candidate_veh_trip_pairs: list):
    #candidate_veh_trip_pairs[veh, req, sche, cost, score]
    for vt_pair_idx in range(len(candidate_veh_trip_pairs)): 
        delay = compute_schedule_delay(candidate_veh_trip_pairs[vt_pair_idx][2], candidate_veh_trip_pairs[vt_pair_idx][0])
        candidate_veh_trip_pairs[vt_pair_idx][4] = delay

def compute_schedule_delay(schedule:list, veh:Veh): #sum(time of nodes to nodes) - sum(shortest time for all trips)
    #schedule = [node[node_id, type, num_people, max_wait_time, shortest_Trip_Time(only for PU node)]], type = 1 for PU, -1 for DO
    total_delay = 0.0
    shortest_time = 0.0
    actual_time = veh.time_to_complete_current_arc
    
    actual_time += compute_schedule_time_cost(schedule) #sum(time of nodes to nodes)
    shortest_time = sum([node[4] for node in schedule if node[1] == 1]) #sum(shortest time for all trips)
    total_delay = actual_time - shortest_time
    return total_delay

def compute_schedule_time_cost(schedule: list): 
    total_schedule_time = 0.0
    for i in range(len(schedule) - 1):
        total_schedule_time += get_timeCost(schedule[i][0], schedule[i+1][0])

    return total_schedule_time

def compute_extra_delay(inserted_schedule: list, insert_position: int, veh: Veh):
    if insert_position == len(inserted_schedule) - 1: #if the inserted node is the last node
        extra_delay = get_timeCost(inserted_schedule[insert_position-1][0], inserted_schedule[insert_position][0]) # use time to insert node as extra delay
        return extra_delay
    elif insert_position == 0: #if the inserted node is the first node
        new_time = get_timeCost(veh.current_node, inserted_schedule[0][0]) + get_timeCost(inserted_schedule[0][0], inserted_schedule[1][0])
        original_time = get_timeCost(veh.current_node, inserted_schedule[1][0]) 
        extra_delay = new_time - original_time
        return extra_delay
    else:
        original_time = get_timeCost(inserted_schedule[insert_position-1][0], inserted_schedule[insert_position+1][0])
        new_time = get_timeCost(inserted_schedule[insert_position-1][0], inserted_schedule[insert_position][0]) + get_timeCost(inserted_schedule[insert_position][0], inserted_schedule[insert_position+1][0])
        extra_delay = new_time - original_time
        return extra_delay

def test_constraints(schedule: list, veh: Veh): 
    # # node[node_id, type, num_people, max_wait_time, shortest_Trip_Time]
    # current_load = veh.load #initial load
    # max_capacity = veh.capacity #initial capacity
    # current_time = veh.veh_time #initial time
    # current_node = veh.current_node #initial node
    # accumulated_time = 0.0 #initial accumulated time
    # for node in schedule:
    #     #1. Capacity
    #     if node[1] == 1: #PU
    #         current_load += node[2] #add the number of people to the current load
    #         if current_load > max_capacity: #if the current load exceeds the vehicle's capacity
    #             return False
    #     elif node[1] == -1: #DO
    #         current_load -= node[2] #subtract the number of people from the capacity
    #         if current_load < 0: #if the capacity is negative
    #             assert "Error: Negative capacity"
    #             return False
    #     #2.1 Max Waiting for Pickup
    #     time_to_next_node = get_timeCost(current_node, node[0]) #time to travel to node
    #     accumulated_time += time_to_next_node #time to travel to node + current time
    #     if node[1] == 1: #PU
    #         time_to_pickup = time_to_next_node #time to travel to node
    #         # max_waiting_time_pickup = node[3]*60 #max wait time in sec
    #         if time_to_pickup > MAX_PICKUP_WAIT_TIME:
    #             return False
    #         current_node = node[0] #update current node
    #     #2.2 Max Waiting for Dropoff
    #     if node[1] == -1: #DO
    #         max_detour_time_dropoff = current_time + node[4] + MAX_DETOUR_TIME 
    #         if accumulated_time > max_detour_time_dropoff: #if the time to travel to node exceeds the max detour
    #             return False
    #         current_time = accumulated_time #update current time
    #         current_node = node[0] #update current node

    # [req.Ori_id, 1, req.Num_people, req.Latest_PU_Time, req.Shortest_TT, req.Req_ID, 'NEW_PU'])
    if veh.target_node == None: #if the vehicle has no target node
        start_node = veh.current_node
        start_time = veh.veh_time
    else:
        start_node = veh.target_node
        start_time = veh.veh_time + veh.time_to_complete_current_arc

    accumulated_time = start_time #initial accumulated time
    current_node = start_node #initial node

    current_load = pickle.loads(pickle.dumps(veh.load)) #initial load
    max_capacity = pickle.loads(pickle.dumps(veh.capacity)) #initial capacity
    # current_time = pickle.loads(pickle.dumps(veh.veh_time)) #initial time
    # current_node = pickle.loads(pickle.dumps(veh.current_node)) #initial node
    
    # time_marks = pickle.loads(pickle.dumps(veh.time_marks_for_visited_nodes)) #time marks for visited nodes of this veh
    for node in schedule:
        
        # time_to_next_node = get_timeCost(current_node, node[0]) #time to travel to node
        time_to_next_node = get_timeCost(current_node, node[0]) #time to travel to node

        # accumulated_time += time_to_next_node #time to travel to node + current time
        # current_time += time_to_next_node #time to travel to node + current time
        accumulated_time += time_to_next_node #time to travel to node + current time

        # time_marks.append([node[5], node[1] ,accumulated_time]) #[req_ID, node_type, accumulated_time when visit this node]

        if node[1] == 1: #PU [req.Ori_id, 1, req.Num_people, req.Latest_PU_Time, req.Shortest_TT, req.Req_ID, 'NEW_PU']
            #1. Capacity
            current_load += node[2] #add the number of people to the current load
            if current_load > max_capacity: #if the current load exceeds the vehicle's capacity
                return False, None
            #2.1 Max Waiting for Pickup
            # if time_to_next_node > MAX_PICKUP_WAIT_TIME:
            #     return False

            #2.1 Max Waiting for Pickup
            if accumulated_time > node[3]:
                return False, None
            current_node = node[0] #update current node

        else: #DO [req.Des_id, -1, req.Num_people, req.Latest_DO_Time, req.Shortest_TT, req.Req_ID, 'NEW_DO']
            #1. Capacity
            current_load -= node[2] #subtract the number of people from the capacity
            if current_load < 0: #if the load is negative
                assert "Error: Negative load"
                return False, None

            #2.2 Max Waiting for Dropoff 
            # PU_time_mark = [mark for mark in time_marks if mark[0] == node[5] and mark[1] == 1][0][2] #find the PU time mark for this DO node
            # travel_time_for_this_pair = accumulated_time - PU_time_mark #time to travel from PU to DO
            # if travel_time_for_this_pair > node[4] : #if the time to travel to node exceeds the max detour+shortest time
            #     return False
            # if accumulated_time > current_time + node[4] + MAX_DETOUR_TIME : #if the time to travel to node exceeds the max detour
            #     return False

            #2.2 Max Waiting for Dropoff 
            if accumulated_time > node[3]:
                return False, None
            current_node = node[0] #update current node

    schedule_time = accumulated_time - start_time
    return True, schedule_time



def insert_request_into_schedule_half(schedule: list, request: Req, PU_node_position: int, DO_node_position):
    if DO_node_position == None: #only PU node
        PU_node = [request.Ori_id, 1, request.Num_people, request.Latest_PU_Time, request.Shortest_TT, request.Req_ID, 'NEW_PU']
        new_schedule = pickle.loads(pickle.dumps(schedule))
        new_schedule_part1 = new_schedule[:PU_node_position]
        new_schedule_part2 = new_schedule[PU_node_position:]
        new_schedule_part1.append(PU_node)
        new_schedule_part1.extend(new_schedule_part2)
        return new_schedule_part1
    
    else:
        DO_node = [request.Des_id, -1, request.Num_people, request.Latest_DO_Time, request.Shortest_TT, request.Req_ID, 'NEW_DO']
        
        # # new_schedule = copy.deepcopy(schedule) 
        # new_schedule =pickle.loads(pickle.dumps(schedule))  #must use deepcopy. otherwise will stuck in infinite loop
        # new_schedule.insert(PU_node_position, PU_node)
        # new_schedule.insert(DO_node_position, DO_node)
        # return new_schedule

        # use extend/append instead of insert, much faster when shcedule is long
        new_schedule = pickle.loads(pickle.dumps(schedule))

        new_schedule_part3 = new_schedule[:DO_node_position]
        new_schedule_part4 = new_schedule[DO_node_position:]
        new_schedule_part3.append(DO_node)
        new_schedule_part3.extend(new_schedule_part4)

        return new_schedule_part3

def insert_request_into_schedule(schedule: list, request: Req, PU_node_position: int, DO_node_position: int):
        PU_node = [request.Ori_id, 1, request.Num_people, request.Latest_PU_Time, request.Shortest_TT, request.Req_ID, 'NEW_PU']
        DO_node = [request.Des_id, -1, request.Num_people, request.Latest_DO_Time, request.Shortest_TT, request.Req_ID, 'NEW_DO']
        
        # # new_schedule = copy.deepcopy(schedule) 
        # new_schedule =pickle.loads(pickle.dumps(schedule))  #must use deepcopy. otherwise will stuck in infinite loop
        # new_schedule.insert(PU_node_position, PU_node)
        # new_schedule.insert(DO_node_position, DO_node)
        # return new_schedule

        # use extend/append instead of insert, much faster when shcedule is long
        new_schedule = pickle.loads(pickle.dumps(schedule))
        new_schedule_part1 = new_schedule[:PU_node_position]
        new_schedule_part2 = new_schedule[PU_node_position:]
        new_schedule_part1.append(PU_node)
        new_schedule_part1.extend(new_schedule_part2)

        new_schedule_part3 = new_schedule_part1[:DO_node_position]
        new_schedule_part4 = new_schedule_part1[DO_node_position:]
        new_schedule_part3.append(DO_node)
        new_schedule_part3.extend(new_schedule_part4)

        return new_schedule_part3
    
def upd_schedule_for_vehicles_in_selected_vt_pairs(candidate_veh_trip_pairs: list,
                                                   selected_veh_trip_pair_indices: List[int]):
    assigned_reqs = []
    for idx in selected_veh_trip_pair_indices:
        #For Simonetto's Method, there is only one req for each trip.
        [veh, req, sche, cost, score] = candidate_veh_trip_pairs[idx]
        assert sche != []
        # Check if sche contains elements with 'NEW_PU' as its last element
        if any(node[-1] == 'NEW_PU' for node in sche):
            # Handle the case where sche contains elements with 'NEW_PU' as its last element
            pass
        else:
            assert KeyError, "Error: sche does not contain elements with 'NEW_PU' as its last element"

        req.Status = OrderStatus.PICKING
        veh.status = VehicleStatus.WORKING
        veh.update_schedule(sche)
        assigned_reqs.append(req)
        veh.assigned_reqs.append(req.Req_ID)
    return assigned_reqs
        