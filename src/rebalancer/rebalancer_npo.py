from src.dispatcher.ilp_assign import *


def reposition_idle_vehicles_to_nearest_pending_orders(reqs: List[Req], vehs: List[Veh]):

    # 1. Get a list of the unassigned orders.
    pending_reqs = [req for req in reqs if req.Status == OrderStatus.PENDING]
    if len(pending_reqs) == 0: #no pending requests
        return

    # if DEBUG_PRINT:
    #     num_of_idle_vehs = 0
    #     for veh in vehs:
    #         if veh.status == VehicleStatus.IDLE:
    #             num_of_idle_vehs += 1
    #     print(f"        -Repositioning {num_of_idle_vehs} idle vehicles to "
    #           f"{len(pending_rids)} locations through NPO...")

    # 2. Compute all rebalancing candidates.
    rebl_veh_req_pairs = []
    for req in pending_reqs:
        for veh in vehs:
            if veh.status != VehicleStatus.IDLE:
                continue
            rebl_timeCost = get_timeCost(veh.current_node, req.Ori_id)
            #schedule = [node[node_id, type, num_people, max_wait_time, shortest_Trip_Time(only for PU node)]], type = 1 for PU, -1 for DO
            rebl_sche = [[req.Ori_id, 1, req.Num_people, req.Max_wait, req.Shortest_TT]] 
            rebl_veh_req_pairs.append([veh, [req], rebl_sche, rebl_timeCost])
            # detour_sec = 240  # A hyper parameter and 240 is probably not the best option.
            #  A rebalancing vehicle is allowed to pick up new orders
            #  if it can still visit the reposition waypoint with a small detour.
            # rebl_sche = [(-1, 0, req.onid, system_time_sec + rebl_dt + detour_sec)]
            # rebl_veh_req_pairs.append([veh, [req], rebl_sche, rebl_dt, -rebl_dt])

    # # 000. Score the rebalancing task using value functions.
    # if len(rebl_veh_req_pairs) > 1:
    #     expected_values = value_func.compute_expected_values_for_veh_trip_pairs(num_of_new_reqs, vehs,
    #                                                                             rebl_veh_req_pairs, system_time_sec)
    #     for idx, vr_pair in enumerate(rebl_veh_req_pairs):
    #         vr_pair[4] = expected_values[idx]

    # 3. Select suitable rebalancing candidates. Greedily from the one with the shortest travel time.
    rebl_veh_req_pairs.sort(key=lambda e: e[3])
    selected_vids = []
    selected_rids = []
    for [veh, [req], sche, rebl_timeCost] in rebl_veh_req_pairs:
        # Check if the vehicle has been selected to do a rebalancing task.
        if veh.id in selected_vids:
            continue
        # Check if the visiting point in the current rebalancing task has been visited.
        if req.Req_ID in selected_rids:
            continue
        selected_vids.append(veh.id)
        selected_rids.append(req.Req_ID)

        # 4. Push the rebalancing task to the assigned vehicle.
        # assert len(sche) < 5 #DEBUG CODE
        veh.update_schedule(sche)
        # 5. Update the vehicle status.
        veh.status = VehicleStatus.REBALANCING

    # if DEBUG_PRINT:
    #     print(f"            +Rebalancing vehicles: {len(selected_vids)} ({timer_end(t)})")