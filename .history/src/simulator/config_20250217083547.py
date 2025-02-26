"""
constants are found here
"""
import pickle
import os
import pandas as pd
from dateutil.parser import parse

##################################################################################
# Changable Attributes
##################################################################################

class ConfigManager:
    def __init__(self):
        self.settings = {
            "REWARD_THETA": 15.0,
            "REWARD_TYPE": 'REJ',# or 'REJ'
            "NODE_LAYERS": 1, # number of layers of rejected rate to consider
            "MOVING_AVG_WINDOW": 40, # 10mins
            "DECAY_FACTOR": 0.9,
            "RL_DURATION": 3600, # The epoch length
            "LEARNING_WINDOW": 1800, # 30 mins
            "CONSIDER_NUM_CYCLES": 4, #num of last cycles to consider in reward calculation, 10 mins
        }
    def get(self, key):
        return self.settings[key]
    def set(self, key, value):
        self.settings[key] = value

        
##################################################################################
# Data File Pathb
##################################################################################
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# MAP_NAME = "SmallGrid" 
# MAP_NAME = "Manhattan"
MAP_NAME = "Utrecht"

if MAP_NAME == "SmallGrid":
    PATH_ALL_PATH_TABLE = f"{ROOT_PATH}/SmallGridData/SmallGrid_AllPathTable.pickle"
    PATH_ARCS = f"{ROOT_PATH}/SmallGridData/SmallGrid_Arcs.csv"
    PATH_REQUESTS = f"{ROOT_PATH}/SmallGridData/SmallGrid_Requests.csv"
    PATH_TIMECOST = f"{ROOT_PATH}/SmallGridData/SmallGrid_TimeCost.csv"
    PATH_NODES_LOOKUP_TABLE = f"{ROOT_PATH}/SmallGridData/SmallGrid_Nodes_Lookup_Table.csv"
    PATH_AREA_ADJ_MATRIX = f"{ROOT_PATH}/SmallGridData/SmallGrid_AREA_Adjacency_Matrix.pickle"
    PATH_TEMP_REQ = f"{ROOT_PATH}/SmallGridData/temp_req.csv"
    NUM_NODES = 100

elif MAP_NAME == "Manhattan":
    PATH_ALL_PATH_MATRIX = f"{ROOT_PATH}/NYC/NYC_Manhattan_AllPathMatrix.pickle"
    PATH_ALL_PATH_TIME_MATRIX = f"{ROOT_PATH}/NYC/Manhattan_AllPathTimeMatrix.pickle"
    PATH_NODE_ADJ_MATRIX = f"{ROOT_PATH}/NYC/Manhattan_Node_Adjacency_Matrix.pickle"
    PATH_ZONE_ADJ_MATRIX = f"{ROOT_PATH}/NYC/Manhattan_Zone_Adjacency_Matrix.pickle"
    PATH_NODES = f"{ROOT_PATH}/NYC/Manhattan_Nodes.pickle"
    NUM_NODES = 4091
    NUM_AREA = 32

elif MAP_NAME == "Utrecht":
    PATH_ALL_PATH_MATRIX = f"{ROOT_PATH}/Utrecht/Utrecht_AllPathMatrix.pkl"
    PATH_ALL_PATH_TIME_MATRIX = f"{ROOT_PATH}/Utrecht/Utrecht_AllPathTimeMatrix.pkl"
    PATH_NODE_ADJ_MATRIX = f"{ROOT_PATH}/Utrecht/Utrecht_Node_Adjacency_Matrix.pkl"
    PATH_ZONE_ADJ_MATRIX = f"{ROOT_PATH}/Utrecht/Utrecht_Zone_Adjacency_Matrix.pkl"
    PATH_NODES = f"{ROOT_PATH}/Utrecht/Utrecht_Nodes.pkl"
    PATH_ISOLATED_NODES = f"{ROOT_PATH}/Utrecht/Utrecht_IsolatedNodes.pkl"
    NUM_NODES = 9616
    NUM_AREA = 32

AREA_IDS = list(range(NUM_AREA))

# # small-grid-data
# PATH_SMALLGRID_ARCS = f"{ROOT_PATH}/SmallGridData/SmallGrid_Arcs.csv"
# PATH_SMALLGRID_REQUESTS = f"{ROOT_PATH}/SmallGridData/SmallGrid_Requests.csv"
# PATH_SMALLGRID_TIMECOST = f"{ROOT_PATH}/SmallGridData/SmallGrid_TimeCost.csv"
# PATH_SMALLGRID_ALL_PATH_TABLE = f"{ROOT_PATH}/SmallGridData/SmallGrid_AllPathTable.pickle"

# NUM_NODES_SMALLGRID = 100

# # Manhattan-data
# PATH_MANHATTAN_ALL_PATH_MATRIX = f"{ROOT_PATH}/NYC/NYC_Manhattan_AllPathMatrix.pickle"
# PATH_MANHATTAN_ALL_PATH_TIME_MATRIX = f"{ROOT_PATH}/NYC/NYC_Manhattan_AllPathTimeMatrix.pickle"
# PATH_MANHATTAN_CITYARC = f"{ROOT_PATH}/NYC/NYC_Manhattan_CityArc.pickle"
# PATH_MANHATTAN_REQUESTS = f"{ROOT_PATH}/NYC/NYC_Manhattan_Requests.csv"
# PATH_MANHATTAN_REQUESTS_COMBINED = f"{ROOT_PATH}/NYC/NYC_Andres_data/combined_file_size3.csv"
# PATH_MANHATTAN_AREA_ADJ_MATRIX = f"{ROOT_PATH}/NYC/NYC_Manhattan_AREA_Adjacency_Matrix.pickle"
# PATH_MANHATTAN_NODES_LOOKUP_TABLE = f"{ROOT_PATH}/NYC/NYC_Manhattan_Nodes_Lookup_Table.csv"
# PATH_TEMP_REQ = f"{ROOT_PATH}/NYC/NYC_Andres_data/temp_req.csv"

# NUM_NODES_MANHATTAN = 4091

# node_lookup_table = pd.read_csv(PATH_MANHATTAN_NODES_LOOKUP_TABLE)
# AREA_IDS = node_lookup_table['zone_id'].unique().astype(int)

##################################################################################
# Reinforcement Learning Config
##################################################################################
RL_STEP_LENGTH = 5 # 2.5 mins, 10 steps
WARM_UP_EPOCHS = 0
WARM_UP_DURATION = 1800 # 30 mins 1800
REWARD_COEFFICIENT = 100 
NUM_FEATURES = 7
BATCH_SIZE = 32
REJ_THRESHOLD = 0.3
MAX_THETA = 60.0
MIN_THETA = 0.0

MAX_THETA_STEP = 5.0 # MULTIPLIER OF ACTION 
PAST_REJ_NUM = 1 # how many past rejections to consider, one is 75s, default 5 mins

NOISE_DECAY_RATE = 0.95

ENV_MODE = 'TRAIN' # or 'TEST'
THETA_MODE = 'STEP' # 'STEP'/'DIRECT'
COST_TYPE = 'USER'# or 'OPERATOR', 'BOTH'
MEMORY_SIZE = 24 # how many states to consider in reward calculation, 48 states = 1 hour
PEAK_HOUR_TEST = False
HALF_REQ = False

MISSING_REWARD_PREDICTOR = False # True if using old model with no reward predictor
SKIP_GNN = False # True if skipping GNN for testing

EVAL_STEPS = 48*1.5 #1.5 hours
REBALANCE_FREQ = 600 # 10 minutes
REBALANCE_SIZE = 50 # number of vehicles to consider for rebalancing

REJ_THRESHOLD_REWARD = 0.5
REJ_THRESHOLD_MULTIPLIER = 2.0 * 0.6
BASE_USER_COST = 9.0 
COST_REWARD_MULTIPLIER = 0.1 * 0.4
REWARD_MULTIPLIER = 1.0
REJ_THRESHOLD = 0.1 #if agent stuck at border but rej_rate lower than this, small reward will be applied.

MAX_REWARD = 1.0
REBALANCE_FREQUENCY = 300 # 5 minutes
REWARD_OFFSET = -0.2

if HALF_REQ:
    REJ_THRESHOLD_REWARD = 0.25 #R_t
    BASE_USER_COST = 2.5 #C_t
    REJ_THRESHOLD_MULTIPLIER = 1.5 #M_r
    COST_REWARD_MULTIPLIER = 0.1 #M_c
    REJ_THRESHOLD = 0.1 #if agent stuck at border but rej_rate lower than this, small reward will be applied.

if MAP_NAME == 'Utrecht':
    REJ_THRESHOLD_REWARD = 0.02
    REJ_THRESHOLD_MULTIPLIER = 2.0 * 0.6
    BASE_USER_COST = 0.3
    COST_REWARD_MULTIPLIER = 0.1 * 0.4
    REWARD_MULTIPLIER = 10.0
    REWARD_OFFSET = 0
    REJ_THRESHOLD = 0.0 #if agent stuck at border but rej_rate lower than this, small reward will be applied.
##################################################################################
# Mod System Config
##################################################################################
# dispatch_config
DISPATCHER = "SBA"        # 3 options: SBA, OSP-NR, OSP
REBALANCER = "NJO"        # 3 options: NONE, NPO, NJO

HEURISTIC_ENABLE = True

# for small-grid-data
# FLEET_SIZE = [100]
# VEH_CAPACITY = [4]
# MAX_PICKUP_WAIT_TIME = 5 # 5 min
# MAX_DETOUR_TIME = 10 # 10 min

# for Manhattan-data
if MAP_NAME == 'Manhattan':
    FLEET_SIZE = [1000] #1000
    VEH_CAPACITY = [3]
elif MAP_NAME == 'Utrecht':
    FLEET_SIZE = [200]
    VEH_CAPACITY = [3]

MAX_PICKUP_WAIT_TIME = 5*60 # 5 min
MAX_DETOUR_TIME = 10*60 # 10 min

MAX_NUM_VEHICLES_TO_CONSIDER = 20
MAX_SCHEDULE_LENGTH = 30

MAX_DELAY_REBALANCE = 10*60 # 10 min

if HALF_REQ:
    PENALTY = 0.2
else:
    PENALTY = 0.9 #penalty for ignoring a request
REBALANCER_PENALTY = 80000.0 #penalty for ignoring a request in rebalancer
MAX_REBALANCE_CONSIDER = 3600

##################################################################################
# Anticipatory ILP Config
##################################################################################
# REWARD_THETA = 0
PW = 4.64/3600 # usd/s User's Cost of waiting
PV = 2.32/3600 # usd/s User's Cost of travelling in vehicle
PO = 3.48/3600 # usd/s Operator's Cost of operating a vehicle
# PO = 0.0 

# REWARD_TYPE = 'GEN' # or 'REJ'
# NODE_LAYERS = 1 # number of layers of rejected rate to consider
PSI = 1 #𝜓 is a tuning parameter (the higher this parameter, the more uniform the resulting rates).
##################################################################################
# Simulation Config
##################################################################################
DEBUG_PRINT = False

# for small-grid-data
# SIMULATION_DURATION = 60.0 #SmallGridData has 60 minutes of data
# TIME_STEP = 0.25 # 15 seconds
# COOL_DOWN_DURATION = 60.0 # 60 minutes
# PENALTY = 5.0 #penalty for ignoring a request

# for Manhattan-data
SIMULATION_DURATION = 3600*20 # 60 minutes = 3600 seconds
TIME_STEP = 15 # 15 seconds
COOL_DOWN_DURATION = 3600 # 20 minutes = 1200 seconds

