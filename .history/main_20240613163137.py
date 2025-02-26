
# start time of initialization
from src.utility.utility_functions import *
from src.simulator.Simulator_platform import Simulator_Platform
import cProfile
from src.simulator.config import ConfigManager
import time
# from src.value_function.value_function import ValueFunction

system_initial_time = 0
print(f"[INFO] Initializing the simulator")


def change_config(config: ConfigManager, args:list):
    for variable in args.keys():
        value = args[variable]
        config.set(variable, value)

def run_sim(args:list):
    start_time = time.time()
    config = ConfigManager()
    change_config(config, args) # change the config based on the args 

    REWARD_THETA = config.get('REWARD_THETA')
    REWARD_TYPE = config.get('REWARD_TYPE')
    NODE_LAYERS = config.get('NODE_LAYERS')
    MOVING_AVG_WINDOW = config.get('MOVING_AVG_WINDOW')

    # print(f"[INFO] Running simulation using dispatcher: {DISPATCHER}")
    # print(f"[INFO] Running simulation using fleet size: {FLEET_SIZE[0]}")
    # print(f"[INFO] Running simulation using vehicle capacity: {VEH_CAPACITY[0]}")
    # print(f"[INFO] Running simulation using rebalancer: {REBALANCER}")
    # print(f"[INFO] Running simulation with MAX_PICKUP_WAIT_TIME: {MAX_PICKUP_WAIT_TIME} seconds")
    # print(f"[INFO] Running simulation with MAX_DETOUR_TIME: {MAX_DETOUR_TIME} seconds")
    # print(f"[INFO] Running simulation with MAX_NUM_VEHICLES_TO_CONSIDER: {MAX_NUM_VEHICLES_TO_CONSIDER}")
    # print(f"[INFO] Running simulation with MAX_SCHEDULE_LENGTH: {MAX_SCHEDULE_LENGTH}")
    print(f"[INFO] Running simulation with reward theta: {REWARD_THETA}")
    print(f"[INFO] Running simulation with reward type: {REWARD_TYPE}")
    print(f"[INFO] Running simulation with node layers: {NODE_LAYERS}")
    print(f"[INFO] Running simulation with moving average window: {MOVING_AVG_WINDOW}")
    platform = Simulator_Platform(system_initial_time,config)
    platform.run_simulation()
    end_time = time.time()
    runtime = end_time - start_time
    platform.create_report(runtime)



if __name__ == '__main__':
    # cProfile.run('run_sim([])', 'runtime.out')
    args = {'REWARD_THETA': 1.0, 'REWARD_TYPE': 'REJ', 'NODE_LAYERS': 2, 'MOVING_AVG_WINDOW': 40}
    run_sim(args) # run the simulation with default values

