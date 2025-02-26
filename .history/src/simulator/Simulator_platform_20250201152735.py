from src.simulator.config import * #defines the constants
import pandas as pd
import numpy as np
import cv2
from src.simulator.types import *
from src.simulator.config import *
from src.simulator.vehicle import Veh
from src.simulator.request import Req
from src.simulator.statistic import Statistic
from src.simulator.route_functions import *
from src.dispatcher.dispatch_sba import *
from src.dispatcher.ilp_assign import *
from src.dispatcher.scheduling import *
from src.rebalancer.rebalancer_ilp_assignment import *
from src.rebalancer.rebalancer_npo import *
from src.rebalancer.rebalancer_njo import *
from src.utility.utility_functions import *
from src.RL.data_processing import *
import logging
from tqdm import tqdm

# ALLPATHTABLE = 'SmallGrid_AllPathTable.pickle'
# NETWORK_NAME = 'SmallGridData'
# BASEDIR = osp.dirname(osp.abspath(__file__))

# with open(osp.join(BASEDIR, '../..', NETWORK_NAME, ALLPATHTABLE), 'rb') as f:
# # with open(BASEDIR + '/' + NETWORK_NAME+ '/' + ALLPATHTABLE, 'rb') as f:
#     all_path_table = pickle.load(f)


class Simulator_Platform(object):
     
    def __init__(self, simulation_start_time_stamp: float, config: ConfigManager, init_type='RANDOM'):
        self.config = config
        RL_DURATION = config.get("RL_DURATION")
        self.statistic = Statistic() #initialize the statistic class, Need to check whether there will be multiple instances of this class
        self.start_time = simulation_start_time_stamp
        self.system_time = self.start_time
        self.end_time = self.start_time + RL_DURATION
        self.recorded_end_time = self.start_time
        self.total_time_step = RL_DURATION // TIME_STEP
        self.accumulated_request = []
        self.rejected_reqs = []
        self.last_req_ID = 0
        # Use deque to achieve moving average. 12 means 3 minutes
        MOVING_AVG_WINDOW = self.config.get("MOVING_AVG_WINDOW")    
        self.num_of_generate_req_for_areas_dict_movingAvg = {i: deque(maxlen=MOVING_AVG_WINDOW) for i in AREA_IDS} # {area_id: generate_req} used in reward calculation
        self.num_of_rejected_req_for_areas_dict_movingAvg = {i: deque(maxlen=MOVING_AVG_WINDOW) for i in AREA_IDS} # {area_id: rejected_req} used in reward calculation
        self.num_of_attraction_for_areas_dict_movingAvg = {i: deque(maxlen=MOVING_AVG_WINDOW) for i in AREA_IDS} # {destination area_id: generate_req} used in reward calculation

        self.RL_step_flag = False
        # Initialize the fleet with random initial positions.
        self.vehs = []
        # Initialize the demand generator.
        self.reqs = []

        self.current_cycle_rej_rate = []
        
        if init_type == 'UNIFORM':
            self.theta = 30.0
        else:
            self.theta = float(np.random.uniform(5,55)) #initialize the theta value for reward calculation

        self.past_rejections = []
        self.past_costs = []
        self.past_operators_costs = []
        self.past_users_costs = []
        self.idle_veh_num_list = []
        self.past_thetas = []
        # if init_type == 'UNIFORM':
        #     self.theta = 0.0
        # else:
        #     self.theta = float(np.random.uniform(5,55))

        
        print("---------------------------------")
        print(f"[INFO] Platform is initialized.")
        print(f"[INFO] Initial Theta: {self.theta}")
        print("---------------------------------")


        if MAP_NAME == 'Manhattan': # [ReqID Oid Did ReqTime Size]
            # print(f"[INFO] Randomly Loading ManhattanData...")
            if init_type == 'UNIFORM': #eval mode
                self.req_loader(0,0, fixed_day=10)
            else:
                self.req_loader(0,0)

            node_id_uniform = list(range(1,NUM_NODES+1,NUM_NODES//FLEET_SIZE[0]))
            for i in range(FLEET_SIZE[0]):
                if init_type == 'UNIFORM':
                    # Initialize the vehicles with uniform initial positions.
                    self.vehs.append(Veh(i, self.system_time, node_id_uniform[i], VEH_CAPACITY[0]))
                elif init_type == 'RANDOM':
                    # Initialize the vehicles with random initial positions.
                    node_id_random = np.random.randint(1, NUM_NODES+1)
                    self.vehs.append(Veh(i, self.system_time, node_id_random, VEH_CAPACITY[0]))
                
            # print(f"[INFO] Vehicles Initialization finished")

        elif MAP_NAME == 'SmallGrid':
            print(f"[INFO] Loading SmallGridData...")
            reqs = pd.read_csv(PATH_SMALLGRID_REQUESTS)
            reqs_data = reqs.to_numpy()
            for i in range(reqs_data.shape[0]):
                self.reqs.append(Req(int(reqs_data[i, 0]), int(reqs_data[i, 1]), float(reqs_data[i, 2]), int(reqs_data[i, 3]), float(reqs_data[i, 4]), int(reqs_data[i, 5])))
            print(f"[INFO] Requests Initialization finished")
            for i in range(FLEET_SIZE[0]):
                node_id = np.random.randint(0, 100)
                self.vehs.append(Veh(i, self.system_time, node_id, VEH_CAPACITY[0]))
            print(f"[INFO] Vehicles Initialization finished")
        
        elif MAP_NAME == 'Utrecht':
            if init_type == 'UNIFORM': #eval mode
                self.req_loader(0,0, fixed_day=10)
            else:
                self.req_loader(0,0)

            node_id_uniform = list(range(1,NUM_NODES+1,NUM_NODES//FLEET_SIZE[0]))
            for i in range(FLEET_SIZE[0]):
                if init_type == 'UNIFORM':
                    # Initialize the vehicles with uniform initial positions.
                    self.vehs.append(Veh(i, self.system_time, node_id_uniform[i], VEH_CAPACITY[0]))
                elif init_type == 'RANDOM':
                    # Initialize the vehicles with random initial positions.
                    node_id_random = np.random.randint(1, NUM_NODES+1)
                    self.vehs.append(Veh(i, self.system_time, node_id_random, VEH_CAPACITY[0]))


        else:
            assert (False and "[ERROR] WRONG MAP NAME SETTING! Please check the name of map in config!")
        
        self.req_init_idx = 1
        self.statistic.total_requests = len(self.reqs)

        # Initialize the dispatcher and the rebalancer.
        if DISPATCHER == "SBA":
            self.dispatcher = DispatcherMethod.SBA
        else:
            assert (False and "[ERROR] WRONG DISPATCHER SETTING! Please check the name of dispatcher in config!")
        if REBALANCER == "NONE":
            self.rebalancer = RebalancerMethod.NONE
        elif REBALANCER == "NPO":
            self.rebalancer = RebalancerMethod.NPO
        elif REBALANCER == "NJO":
            self.rebalancer = RebalancerMethod.NJO
        else:
            assert (False and "[ERROR] WRONG REBALANCER SETTING! Please check the name of rebalancer in config!")

        if HEURISTIC_ENABLE:
            pass
            # print(f"[INFO] Heuristic method is enabled.")
        # print(f"[INFO] Platform is ready. ")


    def run_simulation(self):     
        simulation_end_step = int(self.total_time_step)+1  
        cool_down_step = int(COOL_DOWN_DURATION // TIME_STEP)
        cool_down_flag = True
        for current_time_step in tqdm(range(max(int(self.start_time//TIME_STEP),1), simulation_end_step + cool_down_step, 1), desc=f"Ricardo's Simulator"):
            self.system_time = current_time_step * TIME_STEP
            if cool_down_flag and current_time_step > simulation_end_step:
                print("Cooling down...")
                cool_down_flag = False
            self.run_cycle(cool_down_flag)
        
        self.RL_step_flag = True #set the flag to True to indicate the end of the simulation

    
    def run_cycle(self):
        # 1. Update the vehicles' positions
        self.update_veh_to_time()

        # 2. Pick up requests in the current cycle. add the requests to the accumulated_request list
        current_cycle_requests = self.get_current_cycle_request(self.system_time)

        # 2.1 Prune the accumulated requests. (Assigned requests and rejected requests are removed.)
        self.accumulated_request.extend(current_cycle_requests)
        current_rejected_reqs = self.prune_requests()
        self.rejected_reqs.extend(current_rejected_reqs)
        # print(f"        -Considered current cycle requests: {len(self.accumulated_request)}")
        # self.reject_long_waited_requests()

        # 3. Assign pending orders to vehicles.
        # availiable_vehicels = self.get_availiable_vehicels() #Not making sense. Should consider all vehicles. current full vehicles can be not full in the next cycle
        if self.dispatcher == DispatcherMethod.SBA:
            # anticipatory_cost_sum = assign_orders_through_sba(self.accumulated_request, self.vehs, self.system_time, 
            #                           self.num_of_rejected_req_for_areas_dict_movingAvg, self.num_of_generate_req_for_areas_dict_movingAvg, 
            #                           self.config, self.theta)
            operators_cost_sum, users_cost_sum, num_of_rejected_req_for_areas_dict_movingAvg = assign_orders_through_sba(self.accumulated_request, self.vehs, self.system_time, 
                                      self.num_of_rejected_req_for_areas_dict_movingAvg, self.num_of_generate_req_for_areas_dict_movingAvg, 
                                      self.config, self.theta)
            self.num_of_rejected_req_for_areas_dict_movingAvg = num_of_rejected_req_for_areas_dict_movingAvg
            # print(self.num_of_rejected_req_for_areas_dict_movingAvg)
      
        # 4. Reposition idle vehicles to high demand areas.
        if self.rebalancer == RebalancerMethod.NJO:
            rebalancer_njo(self.rejected_reqs, self.vehs, self.system_time)
        elif self.rebalancer == RebalancerMethod.NPO:
            reposition_idle_vehicles_to_nearest_pending_orders(self.accumulated_request, self.vehs)
        
        # 5. Get current cycle rejection rate
        current_rej_rate = self.get_rej_rate(current_cycle_requests)
        self.current_cycle_rej_rate.append(current_rej_rate)

        self.past_rejections.append(current_rej_rate)
        self.past_costs.append(operators_cost_sum + users_cost_sum)
        self.past_operators_costs.append(operators_cost_sum)
        self.past_users_costs.append(users_cost_sum)
        self.idle_veh_num_list.append(self.get_idle_veh_num())
        self.past_thetas.append(self.theta)
        return operators_cost_sum, users_cost_sum

    def get_idle_veh_num(self):
        idle_veh_num = 0
        for veh in self.vehs:
            if veh.status == VehicleStatus.IDLE:
                idle_veh_num += 1
        return idle_veh_num

    def prune_requests(self):
        if DEBUG_PRINT:
            print("                *Pruning candidate vehicle order pairs...", end=" ")

        req_to_be_pruned = []
        rejected_reqs = []
        for req in self.accumulated_request:
            # # 0. Reject overdue requests.
            # if self.system_time >= req.Latest_PU_Time:
            #     req.Status = OrderStatus.REJECTED
            #     req_to_be_pruned.append(req)
            #     rejected_reqs.append(req)
            #     self.num_of_rejected_req_for_nodes_dict[req.Ori_id] += 1
            # 1. Remove the assigned requests from the accumulated requests.
            if req.Status == OrderStatus.PICKING:
                req_to_be_pruned.append(req)
            # 2. Remove the rejected requests from the accumulated requests.
            elif req.Status == OrderStatus.REJECTED or req.Status == OrderStatus.REJECTED_REBALANCED:
                req_to_be_pruned.append(req)
                rejected_reqs.append(req)

        self.accumulated_request = [req for req in self.accumulated_request if req not in req_to_be_pruned]
        return rejected_reqs
    
    def reject_long_waited_requests(self):
        # Reject the long waited orders.
        for req in self.accumulated_request:
            if req.Status != OrderStatus.PENDING:
                continue
            if self.system_time >= req.Req_time + req.Max_wait or self.system_time >= req.Latest_PU_Time:
                req.Status = OrderStatus.REJECTED
                # self.statistic.total_rejected_requests += 1
                # print(f"Request {req.Req_ID} has been rejected.")
                # print(f"Total Rejected Requests: {self.statistic.total_rejected_requests}")
    
    # update vehs and reqs status to their planned positions at time self.system_time
    def update_veh_to_time(self):
        if DEBUG_PRINT:
            print(f"        -Updating vehicles positions and orders statuses to time {self.system_time}...")

        # update the vehicles' positions to the current time
        for veh in self.vehs:
            veh.move_to_time(self.system_time)


    def get_current_cycle_request(self, current_time ) -> list:
        last_req_ID = self.reqs[-1].Req_ID
        if current_time % 3600 == 0:
            self.req_loader(current_time, last_req_ID)
            
        current_cycle_requests = []

        gen_reqs_per_area = [0] * len(AREA_IDS)
        attraction_per_area = [0] * len(AREA_IDS)

        # gen_reqs_per_areas_dict = {i: 0 for i in AREA_IDS} # {node_id: num_gen_req} used in reward calculation
        # attraction_per_areas_dict = {i: 0 for i in AREA_IDS} # {node_id: num_gen_req} used in reward calculation
        
        for req in self.reqs:
            #EXP: to make it faster. Assume the requests are sorted by Req_time    
            if req.Req_time > current_time:
                break

            elif current_time - TIME_STEP <= req.Req_time:
                current_cycle_requests.append(req)
                ori_area_id = map_node_to_area(req.Ori_id)
                dest_area_id = map_node_to_area(req.Des_id)
                gen_reqs_per_area[ori_area_id] += 1
                gen_reqs_per_area[dest_area_id] += 1
        
        for area_id in AREA_IDS: #0 to 31, 32 areas
            self.num_of_generate_req_for_areas_dict_movingAvg[area_id].append(gen_reqs_per_area[area_id])
            self.num_of_attraction_for_areas_dict_movingAvg[area_id].append(attraction_per_area[area_id])

        self.last_req_ID = current_cycle_requests[-1].Req_ID

        return current_cycle_requests

    def get_availiable_vehicels(self) -> list: #vehicle should has at least 1 capacity
        availiable_vehicels = [veh for veh in self.vehs if veh.load < veh.capacity]
        return availiable_vehicels
    
    def get_all_veh_positions(self):
        veh_positions = [veh.current_node for veh in self.vehs]
        # for veh in self.vehs:
        #     veh_positions.append(veh.current_node)
        return veh_positions

    def create_report(self,runtime):
        # REWARD_THETA = self.config.get("REWARD_THETA")
        REWARD_TYPE = self.config.get("REWARD_TYPE")
        NODE_LAYERS = self.config.get("NODE_LAYERS")
        MOVING_AVG_WINDOW = self.config.get("MOVING_AVG_WINDOW")
        for req in self.reqs:
            if req.Status == OrderStatus.REJECTED or req.Status == OrderStatus.REJECTED_REBALANCED:
                self.statistic.total_rejected_requests += 1
            elif req.Status == OrderStatus.PICKING:
                self.statistic.total_picked_requests += 1
            elif req.Status == OrderStatus.PENDING:
                self.statistic.total_pending_requests += 1
        
        for veh in self.vehs:
            self.statistic.total_veh_run_time += veh.run_time
            self.statistic.served_requests_IDs.extend(veh.served_req_IDs)
        
        # total_served_reqs = len(self.statistic.served_requests_IDs)
        rejection_rate = self.statistic.total_rejected_requests / len(self.reqs)
        # avg_req_shortest_TT = np.mean([req.Shortest_TT for req in self.reqs])
        avg_veh_runtime = self.statistic.total_veh_run_time / len(self.vehs)
        avg_req_runtime = self.statistic.total_veh_run_time / len(self.reqs)

        # wrong_reqs = [req.Req_ID for req in self.reqs if req.Status == OrderStatus.PICKING and req.Req_ID not in self.statistic.served_requests_IDs]
        # self.create_video()
        # print(f"Simulation Report:")
        # print(f"Total Requests: {len(self.reqs)}")
        # print(f"Total Rejected Requests: {self.statistic.total_rejected_requests}")
        # print(f"Total Picked Requests: {self.statistic.total_picked_requests}")
        # print(f"Total Served Requests: {total_served_reqs}")
        # print(f"Total Pending Requests: {self.statistic.total_pending_requests}")
        # print(f"Average Shortest Travel Time: {avg_req_shortest_TT}")
        # print(f"Average Vehicle Runtime: {avg_veh_runtime}")
        # print(f"Average Request Runtime: {avg_req_runtime}")
        # print(f"Rejction Rate: {self.statistic.total_rejected_requests / len(self.reqs)}")
        # print(f"Wrong Requests: {len(wrong_reqs)}")
        # print(f"Reward Theta: {REWARD_THETA}")
        
        # print(f"Video has been created.")
        result = {f"REWARD_THETA:{self.theta},REWARD_TYPE:{REWARD_TYPE},REJECTION_RATE:{rejection_rate},NODE_LAYERS:{NODE_LAYERS},MOVING_AVG_WINDOW:{MOVING_AVG_WINDOW},AVG_VEH_RUNTIME:{avg_veh_runtime},AVG_REQ_RUNTIME:{avg_req_runtime},RUNTIME:{runtime}"}
        self.write_results_to_file(result)

    def write_results_to_file(self, results):
        with open("results.txt", "a") as file:  # Open the file in append mode
            for result in results:
                file.write(f"{result}\n")  # Write each result to a new line

    def mapping_node_to_coordinate(self, node_id):
        map_width = 10 #adjust as needed
        video_width = 1000 #adjust as needed
        scale_factor = video_width / map_width
        return (int(node_id % 10 * scale_factor), int(node_id // 10 * scale_factor))
    
    def interpolate_position(self, vehicle_coordinates, scale_factor):
        interpolated_coordinates = []
        for frame in vehicle_coordinates:
            interpolated_coordinates.append(frame)
            next_frame = vehicle_coordinates[vehicle_coordinates.index(frame) + 1]
            for i in range(1, scale_factor):
                interpolated_frame = []
                for j in range(len(frame)):
                    interpolated_frame.append((int(frame[j][0] + (next_frame[j][0] - frame[j][0]) * i / scale_factor), int(frame[j][1] + (next_frame[j][1] - frame[j][1]) * i / scale_factor)))
                interpolated_coordinates.append(interpolated_frame)
        return interpolated_coordinates

    def create_video(self):
        # Define the size of the map and vehicles
        VIDEO_RESOLUTION = (1000,1000)  # Adjust as needed, pixels
        VEHICLE_SIZE = 10  # Adjust as needed, pixels
        VEHICLE_COLOR = (0,0,200) #Light Red color for vehicles, (Blue, Green, Red) for opencv
        GRID_SIZE = (10, 10)
        GRID_COLOR = (128,128,128)  # Grey color for grid lines
        
        # vehicle_positions = copy.deepcopy(self.statistic.all_veh_position_series)
        vehicle_positions = pickle.loads(pickle.dumps(self.statistic.all_veh_position_series))

        vehicle_coordinates = [] # List of frames, each frame is a list of vehicle coordinates
        for frame in vehicle_positions:
            frame_coordinate = []
            for veh_node_id in frame:
                veh_coordinate = self.mapping_node_to_coordinate(veh_node_id) 
                frame_coordinate.append(veh_coordinate)
            vehicle_coordinates.append(frame_coordinate)
        interpolated_vehicle_coordinates = self.interpolate_position(vehicle_coordinates, 5)

        # Create a VideoWriter object
        # cv2.cv.CV_FOURCC(*'XVID')
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        FILE_TYPE = '.mp4'
        # BASEDIR = osp.dirname(osp.abspath(__file__)) + '/'
        OUTPUT_NAME = str(FLEET_SIZE[0]) + 'Vehs_' + str(VEH_CAPACITY[0]) + 'Ppl_' + str(MAX_PICKUP_WAIT_TIME) + 'MaxWait_' + str(MAX_DETOUR_TIME) + 'MaxDetour_' + REBALANCER + 'Rebalancer_' + DISPATCHER + 'Dispatcher' + FILE_TYPE

        # fourcc = cv2.VideoWriter_fourcc(*'avc1')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_NAME, fourcc, 60.0, VIDEO_RESOLUTION)
        # Draw frames and write to video
        for frame_coordinates in interpolated_vehicle_coordinates:
            # Create a blank image representing the map
            frame = np.zeros((VIDEO_RESOLUTION[1], VIDEO_RESOLUTION[0], 3), dtype=np.uint8)
            # Draw grid lines
            for i in range(1, GRID_SIZE[0]):
                cv2.line(frame, (i * (VIDEO_RESOLUTION[0] // GRID_SIZE[0]), 0), (i * (VIDEO_RESOLUTION[0] // GRID_SIZE[0]), VIDEO_RESOLUTION[1]), GRID_COLOR, 1)
            for i in range(1, GRID_SIZE[1]):
                cv2.line(frame, (0, i * (VIDEO_RESOLUTION[1] // GRID_SIZE[1])), (VIDEO_RESOLUTION[0], i * (VIDEO_RESOLUTION[1] // GRID_SIZE[1])), GRID_COLOR, 1)

            # Draw vehicles at their positions
            for vehicle_coordinate in frame_coordinates:
                cv2.circle(frame, tuple(vehicle_coordinate), VEHICLE_SIZE, VEHICLE_COLOR, -1)

            # Write frame to video
            out.write(frame)

        # Release the VideoWriter
        out.release()

    def reset_veh_time(self):
        for veh in self.vehs:
            veh.veh_time = 0

    def req_loader(self, current_sim_time, last_req_ID, fixed_day=None):

        if MAP_NAME == ''
        # PATH_REQUESTS = f"{ROOT_PATH}/NYC/NYC_Andres_data/"
        PATH_REQUESTS = f"{ROOT_PATH}/NYC/Yellow_Taxi_Data/2016_01/"
        if HALF_REQ:
            if PEAK_HOUR_TEST:
                FILE_NAME = "NYC_Manhattan_PeakHour_Requests_half.csv"
                print(f"[INFO] Loading ManhattanData: PeakHour Half")
            else:
                FILE_NAME = "NYC_Manhattan_Requests_half_size3_day"
        else:
            if PEAK_HOUR_TEST:
                FILE_NAME = "NYC_Manhattan_PeakHour_Requests.csv"
                print(f"[INFO] Loading ManhattanData: PeakHour")
            else:
                # FILE_NAME = "NYC_Manhattan_Requests_size3_day"
                FILE_NAME = "NYC_full_day_"
        
        if ENV_MODE == 'TEST':
            # get day one by one
            day = current_sim_time/3600
            day = day%10 + 1 # iterate in [1,10]
            SELECTED_FILE = FILE_NAME + str(int(day)) + '.csv' 
        else:
            if fixed_day:
                day = fixed_day
                SELECTED_FILE = FILE_NAME + str(day) + '.csv'
            else:
                # day = np.random.randint(1, 10)
                # day = np.random.randint(1, 11)
                day = np.random.randint(1, 31)
                # day = 1
                SELECTED_FILE = FILE_NAME + str(day) + '.csv'
            # TEMP_FILE_NAME = 'temp_req.csv'

        with open(os.path.join(PATH_REQUESTS, SELECTED_FILE), 'r') as f:
            temp_req_matrix = pd.read_csv(f)
        temp_req_matrix['ReqID'] += last_req_ID
        temp_req_matrix['ReqTime'] += current_sim_time

        reqs_data = temp_req_matrix.to_numpy()
        for i in range(reqs_data.shape[0]):
            if reqs_data[i,4] > 3:
                continue
            self.reqs.append(Req(int(reqs_data[i, 0]), int(reqs_data[i, 1]), int(reqs_data[i, 2]), int(reqs_data[i, 3]), int(reqs_data[i,4])))
        if not PEAK_HOUR_TEST:
            if HALF_REQ:
                print(f"[INFO] Loading ManhattanData: Half day{day}")
            else:
                print(f"[INFO] Loading ManhattanData: Full day{day}")
#--------------------------------------------------------------------------------
# Below are RL related functions

    def get_simulator_state(self,):
        with open(PATH_NODE_ADJ_MATRIX, 'rb') as a:
            network = pickle.load(a)

        idle_veh_nodes = [0] * (NUM_NODES)
        rebalancing_veh_nodes = [0] * (NUM_NODES)
        current_working_veh_nodes = [0] * (NUM_NODES)
        future_working_veh_nodes_in_5 = [0] * (NUM_NODES)
        future_working_veh_nodes_in_10 = [0] * (NUM_NODES)

        for veh in self.vehs:
            if veh.status == VehicleStatus.IDLE:
                idle_veh_nodes[veh.current_node-1] += 1
            elif veh.status == VehicleStatus.REBALANCING:
                rebalancing_veh_nodes[veh.current_node-1] += 1
            elif veh.status == VehicleStatus.WORKING:
                current_working_veh_nodes[veh.current_node-1] += 1
                # get the node where the vehicle will be in 5 minutes
                time_to_end_route = get_timeCost(veh.current_node, veh.route[-1])
                if time_to_end_route < 5*60: # if the vehicle will reach the destination in 5 minutes, assume veh will stay at the destination
                    future_working_veh_nodes_in_5[veh.route[-1]-1] += 1
                elif time_to_end_route < 10*60: # if the vehicle will reach the destination in 10 minutes, use the node where the vehicle will be in 5 minutes
                    node_in_5 = veh.get_node_in_time(5.0)
                    future_working_veh_nodes_in_5[node_in_5-1] += 1
                else: # veh will not reach the destination in 10 minutes, use the node where the vehicle will be in 10 minutes
                    node_in_10 = veh.get_node_in_time(10.0)
                    future_working_veh_nodes_in_10[node_in_10-1] += 1
        
        gen_counts_nodes = self.num_of_generate_req_for_areas_dict_movingAvg
        rej_counts_nodes = self.num_of_rejected_req_for_areas_dict_movingAvg

        state = [idle_veh_nodes, rebalancing_veh_nodes, current_working_veh_nodes, future_working_veh_nodes_in_5, future_working_veh_nodes_in_10, gen_counts_nodes, rej_counts_nodes]
        state = feature_preparation(state)
        return state, network
    
    def get_simulator_network_by_areas(self,):
        with open(PATH_ZONE_ADJ_MATRIX, 'rb') as a:
            network = pickle.load(a)
        return network
    
    def get_simulator_state_by_areas(self,):
        avaliable_veh_areas = [0] * (NUM_AREA)
        current_working_veh_areas = [0] * (NUM_AREA)
        future_working_veh_areas_in_5 = [0] * (NUM_AREA)
        future_working_veh_areas_in_10 = [0] * (NUM_AREA)

        for veh in self.vehs:
            if veh.status == VehicleStatus.IDLE or veh.status == VehicleStatus.REBALANCING:
                target_area = map_node_to_area(veh.current_node)
                avaliable_veh_areas[target_area] += 1
            elif veh.status == VehicleStatus.WORKING:
                target_area = map_node_to_area(veh.current_node)
                current_working_veh_areas[target_area] += 1

                # get the node where the vehicle will be in future
                time_to_end_route = get_timeCost(veh.current_node, veh.route[-1])
                if time_to_end_route < 5*60: # if the vehicle will reach the destination in 5 minutes, assume veh will stay at the destination
                    target_area = map_node_to_area(veh.route[-1])
                    future_working_veh_areas_in_5[target_area] += 1
                elif time_to_end_route < 10*60: # if the vehicle will reach the destination in 10 minutes, use the node where the vehicle will be in 5 minutes
                    node_in_5, capacity_in_5 = veh.get_node_and_capacity_in_time(5.0)
                    if capacity_in_5 > 0:
                        target_area = map_node_to_area(node_in_5)
                        future_working_veh_areas_in_5[target_area] += 1
                else: # veh will not reach the destination in 10 minutes, use the node where the vehicle will be in 10 minutes
                    node_in_10, capacity_in_10 = veh.get_node_and_capacity_in_time(10.0)
                    if capacity_in_10 > 0:
                        target_area = map_node_to_area(node_in_10)
                        future_working_veh_areas_in_10[target_area] += 1
        
        gen_counts_areas = self.num_of_generate_req_for_areas_dict_movingAvg
        rej_counts_areas = self.num_of_rejected_req_for_areas_dict_movingAvg
        attraction_counts_areas = self.num_of_attraction_for_areas_dict_movingAvg

        state = [avaliable_veh_areas, current_working_veh_areas, future_working_veh_areas_in_5, future_working_veh_areas_in_10 , gen_counts_areas, rej_counts_areas, attraction_counts_areas]
        state = feature_preparation(state)
        
        return state

    def get_rej_rate(self, current_reqs):
        total_rejected_requests = 0
        total_picked_requests = 0
        total_pending_requests = 0
        # To handle the case where no requests are picked up
        for req in current_reqs:
            if req.Status == OrderStatus.REJECTED or req.Status == OrderStatus.REJECTED_REBALANCED:
                total_rejected_requests += 1
            elif req.Status == OrderStatus.PICKING:
                total_picked_requests += 1
            elif req.Status == OrderStatus.PENDING:
                total_pending_requests += 1

        assert(total_pending_requests == 0)        
        
        current_cycle_rejection_rate = total_rejected_requests/(total_picked_requests+total_rejected_requests)
        return current_cycle_rejection_rate

    def is_warm_up_done(self):
        if self.system_time >= WARM_UP_DURATION:
            return True
        else:
            return False
        
    def is_done(self):
        RL_DURATION = self.config.get("RL_DURATION")
        if self.system_time >= self.end_time:
            self.end_time += RL_DURATION
            return True
        else:
            return False
    
    def uniform_reset_simulator(self,):
        self.start_time = 0
        self.__init__(self.start_time, self.config, 'UNIFORM')
    
    def random_reset_simulator(self,):
        # self.start_time = np.random.randint(0,2400) #randomly choose a start time between 0 and 2400
        self.start_time = 0
        self.__init__(self.start_time, self.config, 'RANDOM')
        

    def update_theta(self,new_theta):
        old_theta = self.theta
        # new_theta = old_theta + action
        new_theta = np.clip(new_theta, MIN_THETA, MAX_THETA)
        # self.config.set("REWARD_THETA", new_theta)
        self.theta = new_theta
        if ENV_MODE != 'TEST':
            print(f"Updating REWARD_THETA from {old_theta} to {new_theta}")

#     def graph_to_data(self, network):
#         # Convert the network to edge_index
#         if isinstance(network, pd.DataFrame):
#             network_array = network.values
#         else:
#             network_array = network

#         # Find the indices of non-zero elements, which correspond to edges
#         src_nodes, dst_nodes = np.nonzero(network_array)
#         # edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

#         # 合并成一个 numpy 数组
#         edges = np.vstack((src_nodes, dst_nodes))

#         # 转换为 PyTorch 张量
#         edge_index = torch.tensor(edges, dtype=torch.long)

#         edge_index = edge_index.to(device)
#         # print(edge_index)
        
#         # Ensure the result is in the correct shape (2, num_edges)
#         return edge_index
        
#     def RL_run_cycle(self):
#         # 0. Get the state of the simulator
#         state = self.get_simulator_state_by_areas()

#         # 1. Update the vehicles' positions
#         self.update_veh_to_time()

#         # 2. Pick up requests in the current cycle. add the requests to the accumulated_request list
#         current_cycle_requests = self.get_current_cycle_request(self.system_time)

#         # 2.1 Prune the accumulated requests. (Assigned requests and rejected requests are removed.)
#         self.accumulated_request.extend(current_cycle_requests)
#         current_rejected_reqs = self.prune_requests()
#         self.rejected_reqs.extend(current_rejected_reqs)

#         # 3. Assign pending orders to vehicles.
#         if self.dispatcher == DispatcherMethod.SBA:
#             assign_orders_through_sba(self.accumulated_request, self.vehs, self.system_time, 
#                                       self.num_of_rejected_req_for_areas_dict_movingAvg, self.num_of_generate_req_for_areas_dict_movingAvg, 
#                                       self.config, self.theta)
      
#         # 4. Reposition idle vehicles to high demand areas.
#         if self.rebalancer == RebalancerMethod.NJO:
#             rebalancer_njo(self.rejected_reqs, self.vehs, self.system_time)
#         elif self.rebalancer == RebalancerMethod.NPO:
#             reposition_idle_vehicles_to_nearest_pending_orders(self.accumulated_request, self.vehs)
        
#         # 5. Get current cycle rejection rate
#         self.current_cycle_rej_rate.append(self.get_rej_rate(current_cycle_requests))

#         # 6. Get the next_state of the simulator
#         next_state = self.get_simulator_state_by_areas()

#         # 7. Use the state to get action from the agent
#         # Prepare the RL test environment
#         self.network = self.get_simulator_network_by_areas()
#         self.edge_index = self.graph_to_data(self.network)
        
#         self.fixed_normalizer = FixedNormalizer()

# class FixedNormalizer(object):
#     def __init__(self):
#         path = 'mean_std_data_100800.npz'
#         assert os.path.exists(path), "mean_std_data_100800.npz not found"
#         data = np.load(path)
#         self.mean = torch.tensor(data['mean'], dtype=torch.float32).to(device)
#         self.std = torch.tensor(data['std'], dtype=torch.float32).to(device)
#         self.std[self.std == 0] = 1.0 # avoid division by zero

#     def __call__(self, x):
#         return (x - self.mean) / (self.std)



        