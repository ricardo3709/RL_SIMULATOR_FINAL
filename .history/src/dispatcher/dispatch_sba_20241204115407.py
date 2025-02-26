"""
single request batch assignment, where requests cannot be combined in the same interval
"""

from src.dispatcher.ilp_assign import *
from src.utility.utility_functions import *
from src.simulator.Simulator_platform import *

def assign_orders_through_sba(current_cycle_requests: List[Req], vehs: List[Veh], system_time: float, num_of_rejected_req_for_areas_dict_movingAvg: dict, num_of_generate_req_for_nodes_dict_movingAvg: dict, config: ConfigManager, reward_theta: float):
    if DEBUG_PRINT:
        print(f"        -Assigning {len(current_cycle_requests)} orders to vehicles through SBA...")

    # 1. Compute all possible veh-req pairs, each indicating that the request can be served by the vehicle.
    candidate_veh_req_pairs, considered_vehs = compute_candidate_veh_req_pairs(current_cycle_requests, vehs, system_time)
    # 1.1 Remove identical vehs in considered_vehs
    considered_vehs = list(set(considered_vehs))

    # 2. Score the candidate veh-req pairs. 
    # score_vt_pair_with_delay(candidate_veh_req_pairs)

    # 3. Compute the assignment policy, indicating which vehicle to pick which request.

    #3.1 Pruning the candidate veh-req pairs. (Empty Assign)
    # candidate_veh_req_pairs = prune_candidate_veh_req_pairs(candidate_veh_req_pairs)
    # sort veh/req/pairs
    candidate_veh_req_pairs.sort(key=lambda x: x[3])
    current_cycle_requests.sort(key=lambda x: x.Req_ID)
    considered_vehs.sort(key=lambda x: x.id)
    # selected_veh_req_pair_indices, anticipatory_cost_sum = ilp_assignment(candidate_veh_req_pairs, current_cycle_requests, considered_vehs, 
    #                                                num_of_rejected_req_for_areas_dict_movingAvg, 
    #                                                num_of_generate_req_for_nodes_dict_movingAvg, config, reward_theta)
    selected_veh_req_pair_indices, operators_cost_sum, users_cost_sum = ilp_assignment(candidate_veh_req_pairs, current_cycle_requests, considered_vehs, 
                                                   num_of_rejected_req_for_areas_dict_movingAvg, 
                                                   num_of_generate_req_for_nodes_dict_movingAvg, config, reward_theta)
    

    # 000. Convert and store the vehicles' states at current epoch and their post-decision states as an experience.
    # if COLLECT_DATA and verify_the_current_epoch_is_in_the_main_study_horizon(system_time_sec):
    #     value_func.store_vehs_state_to_replay_buffer(len(new_received_rids), vehs,
    #                                                  candidate_veh_req_pairs, selected_veh_req_pair_indices,
    #                                                  system_time_sec)

    # 4. Update the assigned vehicles' schedules and the assigned requests' statuses.
    assigned_reqs = upd_schedule_for_vehicles_in_selected_vt_pairs(candidate_veh_req_pairs, selected_veh_req_pair_indices)
    # 5. Immediate reject all request that are not assigned.
    immediate_reject_unassigned_requests(current_cycle_requests, assigned_reqs, num_of_rejected_req_for_areas_dict_movingAvg, config)


    # if DEBUG_PRINT:
    #     num_of_assigned_reqs = 0
    #     for rid in current_cycle_requests:
    #         if reqs[rid].status == OrderStatus.PICKING:
    #             num_of_assigned_reqs += 1
    #     print(f"            +Assigned orders: {num_of_assigned_reqs} ({timer_end(t)})")

    return operators_cost_sum, users_cost_sum
    # return anticipatory_cost_sum


def compute_candidate_veh_req_pairs(current_cycle_requests: List[Req], vehs: List[Veh], system_time: float) \
        -> List[Tuple[Veh, List[Req], List[Tuple[int, int, int, float]], float, float]]:    
    if DEBUG_PRINT:
        print("                *Computing candidate vehicle order pairs...", end=" ")

    # Lists to store results
    # Each veh_req_pair = [veh, trip, sche, cost, score]
    candidate_veh_req_pairs = []
    considered_vehs = []

    # Precompute vehicle current nodes and loads to avoid repeated attribute access
    veh_info = [(veh, veh.current_node, veh.capacity - veh.load) for veh in vehs]

    # Process each request in the current cycle
    for req in current_cycle_requests:
        # Filter vehicles that meet basic requirements (time and capacity constraints)
        available_veh = []
        latest_pickup = req.Latest_PU_Time
        req_people = req.Num_people
        req_ori = req.Ori_id

        for veh, current_node, available_capacity in veh_info:
            # Skip vehicles that can't meet time or capacity constraints
            if available_capacity < req_people:
                continue
                
            time_to_origin = get_timeCost(current_node, req_ori)
            if time_to_origin + system_time > latest_pickup:
                continue
                
            available_veh.append([veh, time_to_origin])

        # No feasible vehicles found for this request
        if not available_veh:
            continue

        if HEURISTIC_ENABLE:
            # Apply heuristic: consider only closest vehicles if too many are available
            if len(available_veh) > MAX_NUM_VEHICLES_TO_CONSIDER:
                # Sort by travel time and take top K vehicles
                available_veh.sort(key=lambda x: x[1])
                available_veh = available_veh[:MAX_NUM_VEHICLES_TO_CONSIDER]
                
            # Extract vehicles and add to considered list
            vehicles_only = [pair[0] for pair in available_veh]
            considered_vehs.extend(vehicles_only)
        else:
            # When heuristic is disabled, use all available vehicles
            vehicles_only = [pair[0] for pair in available_veh]

        # Compute best schedule for each feasible vehicle
        for veh in vehicles_only:
            best_sche, cost = compute_schedule(veh, req)
            if best_sche:  # Only add if feasible schedule exists
                candidate_veh_req_pairs.append([veh, req, best_sche, cost, 0.0])
    
    # Remove duplicates from considered vehicles while maintaining order
    if considered_vehs:
        considered_vehs = list(dict.fromkeys(considered_vehs))
        
    return candidate_veh_req_pairs, considered_vehs

def prune_candidate_veh_req_pairs(candidate_veh_req_pairs):
    if DEBUG_PRINT:
        print("                *Pruning candidate vehicle order pairs...", end=" ")

    # for vt_pair in candidate_veh_req_pairs:
    #     veh, req, sche, cost, score = vt_pair
    #     # 1. Remove the assigned requests from the candidate list.
    #     if req.Status == OrderStatus.PICKING:
    #         candidate_veh_req_pairs.remove(vt_pair)
    #     # 2. Remove the rejected requests from the candidate list.
    #     elif req.Status == OrderStatus.REJECTED:
    #         candidate_veh_req_pairs.remove(vt_pair)

    pruned_vq_pair = []
    for vq_pair in candidate_veh_req_pairs:
        veh, req, sche, cost, score = vq_pair
        if req == None: #prune the empty assign option
            continue
        pruned_vq_pair.append(vq_pair)

    return pruned_vq_pair

def immediate_reject_unassigned_requests(current_cycle_requests: List[Req], assigned_reqs: List[Req], num_of_rejected_req_for_areas_dict_movingAvg,config: ConfigManager):
    if DEBUG_PRINT:
        print("                *Immediate rejecting unassigned orders...", end=" ")
    # REWARD_TYPE = config.get('REWARD_TYPE')
    rej_req_per_area = [0] * len(AREA_IDS)

    for req in current_cycle_requests:
        if req not in assigned_reqs:
            req.Status = OrderStatus.REJECTED
            rej_node_id = req.Ori_id
            if REWARD_TYPE == 'REJ':
                area_id = map_node_to_area(rej_node_id)
                rej_req_per_area[target_area_id] += 1

        for area_id in AREA_IDS:
            num_of_rejected_req_for_areas_dict_movingAvg[area_id].append(rej_req_per_area[area_id])

    rej_req_per_area = [0] * len(AREA_IDS)
    for req in current_cycle_requests:
        target_area_id = map_node_to_area(req.Des_id)
        rej_req_per_area[target_area_id] += 1
    for area_id in AREA_IDS:
        num_of_rejected_req_for_areas_dict_movingAvg[area_id].append(rej_req_per_area[area_id])