from src.simulator.Simulator_platform import *
from src.rebalancer.rebalancer_ilp_assignment import *
import random

# rebalance vehicles to area of nearst rejected orders
def rebalancer_njo(reqs: List[Req], vehs: List[Veh], system_time: float):
    if DEBUG_PRINT: 
        print(f"    *Rebalancing idle vehicles to area of nearst rejected orders", end=" ")

    # get rejected requests from last 1 hour
    rejected_reqs = [req for req in reqs if req.Status == OrderStatus.REJECTED and req.Req_time > system_time - MAX_REBALANCE_CONSIDER]
    # rejected_reqs = [req for req in reqs if req.Status == OrderStatus.REJECTED]
    if len(rejected_reqs) == 0: #no rejected requests
        return
    elif len(rejected_reqs) > REBALANCE_SIZE*4: #rejected req more than 200
        indices = np.linspace(0, len(rejected_reqs)-1, REBALANCE_SIZE*4, dtype=int)
        selected_rej_reqs = (rejected_reqs[i] for i in indices)
        rejected_reqs = list(selected_rej_reqs)

    
    total_avaliable_vehs = get_rebalancing_vehs(vehs, system_time)
    # # get idle vehicles
    # total_avaliable_vehs = [veh for veh in vehs if veh.status == VehicleStatus.IDLE]

    # if len(total_avaliable_vehs) < REBALANCE_SIZE: #idle veh less than 50
    #     if system_time % REBALANCE_FREQ == 0: # resupply with rebalancing veh every 10 minutes
    #         rebalancing_vehs = [veh for veh in vehs if veh.status == VehicleStatus.REBALANCING]
    #         if len(rebalancing_vehs) == 0: #no rebalancing vehicles
    #             if len(total_avaliable_vehs) == 0: #no idle vehicles
    #                 return
    #             else: # reblance veh == 0, idle veh > 0, use all idle vehs
    #                 total_avaliable_vehs = total_avaliable_vehs

    #         else: # rebalancing vehicles > 0
    #             sample_size = min(REBALANCE_SIZE - len(total_avaliable_vehs), len(rebalancing_vehs))
    #             # uniform sampling to avoid randomness
    #             step = len(rebalancing_vehs) / sample_size
    #             indices = [int(i * step) for i in range(sample_size)]
    #             sampled_rebalancing_vehs = [rebalancing_vehs[i] for i in indices]
    #             sampled_rebalancing_vehs.extend(total_avaliable_vehs)
    #             total_avaliable_vehs = sampled_rebalancing_vehs # rename total_avaliable_vehs
    #     else:
    #         total_avaliable_vehs = total_avaliable_vehs # use all idle vehs
        
    # 1. Compute all possible veh-req pairs, each indicating that the request can be served by the vehicle.

    if total_avaliable_vehs is None: #no idle vehicles
        return
    else:
        candidate_veh_req_pairs, considered_vehs = compute_candidate_veh_req_pairs(rejected_reqs, total_avaliable_vehs, system_time)
        # 1.1 Remove identical vehs in considered_vehs
        considered_vehs = list(set(considered_vehs))

        # 2. Compute the assignment policy, indicating which vehicle to pick which request.
        selected_veh_req_pair_indices = rebalancer_ilp_assignment(candidate_veh_req_pairs, rejected_reqs, considered_vehs)

        # 3. Update the assigned vehicles' schedules and the assigned requests' statuses.
        rebalanced_vehicles = upd_schedule_for_vehicles_in_selected_vt_pairs(candidate_veh_req_pairs, selected_veh_req_pair_indices)

        # 4. Remove the assigned vehicles from the total_avaliable_vehs
        total_avaliable_vehs = [veh for veh in total_avaliable_vehs if veh not in rebalanced_vehicles]

def compute_candidate_veh_req_pairs(reqs: List[Req], vehs: List[Veh], system_time: float) \
        -> List[Tuple[Veh, List[Req], List[Tuple[int, int, int, float]], float, float]]:    
    candidate_veh_req_pairs = []
    considered_vehs = []
    
    if HEURISTIC_ENABLE:
        # 预先计算所有车辆的当前节点，避免在循环中重复访问
        veh_nodes = [(veh, veh.current_node) for veh in vehs]
        
        for req in reqs:
            # 直接在列表推导中完成筛选和排序，避免中间列表
            available_veh = sorted(
                ((veh, get_timeCost(node, req.Ori_id))
                 for veh, node in veh_nodes
                 if get_timeCost(node, req.Ori_id) <= MAX_DELAY_REBALANCE),
                key=lambda x: x[1]
            )[:MAX_NUM_VEHICLES_TO_CONSIDER]
            
            if not available_veh:  # 使用 not 替代 len() == 0
                continue
            
            # 直接将车辆添加到 considered_vehs
            new_vehs = [v[0] for v in available_veh]
            considered_vehs.extend(new_vehs)
            
            # 计算调度方案
            for veh in new_vehs:
                best_sche, cost = compute_schedule(veh, req)
                candidate_veh_req_pairs.append([veh, req, best_sche, cost, 0.0])
    else:
        considered_vehs = vehs
        for req in reqs:
            for veh in vehs:
                best_sche, cost = compute_schedule(veh, req)
                candidate_veh_req_pairs.append([veh, req, best_sche, cost, 0.0])

    # 当没有找到匹配时尝试所有车辆
    if not candidate_veh_req_pairs:
        considered_vehs = vehs
        for req in reqs:
            for veh in vehs:
                best_sche, cost = compute_schedule(veh, req)
                candidate_veh_req_pairs.append([veh, req, best_sche, cost, 0.0])
        
    return candidate_veh_req_pairs, considered_vehs

def compute_schedule(veh: Veh, req: Req):
    # best_schedule = None
    # min_time_cost = np.inf
    # veh.schedule = veh.remove_duplicate_sublists(veh.schedule)

    # current_schedule = copy.deepcopy(veh.schedule) 
    # current_schedule = pickle.loads(pickle.dumps(veh.schedule))

    # assert current_schedule != None #DEBUG CODE, current_schedule should be None
    # assert len(current_schedule) < 5 #DEBUG CODE

    # [Rebalance_target_node, rebalance_node_type, Fake_num_people, fake_latest_pu_time, rebalancer_req_ID]
    current_schedule = [[req.Ori_id, 0, req.Num_people, req.Latest_PU_Time, req.Shortest_TT, 0]] # 0 means rebalance schedule
    time_cost = get_timeCost(veh.current_node, req.Ori_id)

    return current_schedule, time_cost


def compute_schedule_time_cost(schedule: list): 
    total_schedule_time = 0.0
    for i in range(len(schedule) - 1):
        total_schedule_time += get_timeCost(schedule[i][0], schedule[i+1][0])

    return total_schedule_time


def insert_request_into_schedule(schedule: list, request: Req, PU_node_position: int, DO_node_position: int):
    PU_node = [request.Ori_id, 1, request.Num_people, request.Latest_PU_Time, request.Shortest_TT]
    DO_node = [request.Des_id, -1, request.Num_people, request.Latest_DO_Time, request.Shortest_TT]
    # new_schedule = copy.deepcopy(schedule)
    new_schedule = pickle.loads(pickle.dumps(schedule))
    new_schedule.insert(PU_node_position, PU_node)
    new_schedule.insert(DO_node_position, DO_node)
    return new_schedule


def upd_schedule_for_vehicles_in_selected_vt_pairs(candidate_veh_trip_pairs: list,
                                                   selected_veh_trip_pair_indices: List[int]):
    # assigned_reqs = []
    rebalanced_vehs = []
    for idx in selected_veh_trip_pair_indices:
        #For Simonetto's Method, there is only one req for each trip.
        [veh, req, sche, cost, score] = candidate_veh_trip_pairs[idx]
        veh.update_schedule(sche)
        veh.status = VehicleStatus.REBALANCING
        req.Status = OrderStatus.REJECTED_REBALANCED
        rebalanced_vehs.append(veh)
        # assigned_reqs.append(req)
    # return assigned_reqs
    return rebalanced_vehs

def get_rebalancing_vehs(vehs, system_time):
    # 使用单次遍历同时获取 idle 和 rebalancing 车辆
    idle_vehs = []
    rebalancing_vehs = []
    for veh in vehs:
        if veh.status == VehicleStatus.IDLE:
            idle_vehs.append(veh)
        elif veh.status == VehicleStatus.REBALANCING:
            rebalancing_vehs.append(veh)
    
    if len(idle_vehs) >= REBALANCE_SIZE:
        return idle_vehs[:REBALANCE_SIZE]  # 直接返回需要数量的空闲车辆
        
    # 只在需要补充且到达重平衡时间时处理
    if system_time % REBALANCE_FREQ == 0:
        if not rebalancing_vehs:  # 没有重平衡车辆
            return idle_vehs if idle_vehs else None
            
        # 计算需要补充的数量
        needed = REBALANCE_SIZE - len(idle_vehs)
        sample_size = min(needed, len(rebalancing_vehs))
        
        # 优化采样计算
        if sample_size == len(rebalancing_vehs):
            # 如果需要所有重平衡车辆，直接合并
            idle_vehs.extend(rebalancing_vehs)
        else:
            indices = np.linspace(0, len(rebalancing_vehs)-1, sample_size, dtype=int)
            idle_vehs.extend(rebalancing_vehs[i] for i in indices)
                
        return idle_vehs
    
    return idle_vehs