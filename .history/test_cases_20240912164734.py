import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import sys
import copy
import argparse
import random

import torch
import torch.nn as nn

from joblib import Parallel, delayed

import src.RL.utils as utils
import src.RL.TD3 as TD3
from src.RL.environment import ManhattanTrafficEnv
from src.simulator.config import *
# Define the args
kwargs = {
    "state_dim": 512,
    "action_dim": 1,
    "max_action": 1,
    "discount": 0.99,
    "tau": 0.005,
    "policy_noise": 0.2 * 0.5,
    "noise_clip": 0.5 * 0.5,
    "policy_freq": 2,
}

epoch = 60
model_name = f'TD3_AUTOENCODER_{epoch}'

# Load Policy
policy = TD3.TD3(**kwargs)
policy.load(f"saved_models/{model_name}")

def toggle_theta_test_non_policy(env, toggle_time, toggle_duration, sim_duration, origin_theta, toggle_theta):
    env.simulator.theta = origin_theta
    for _ in tqdm(range(int(sim_duration/TIME_STEP)),desc='Toggle Theta Test'):
        env.simulator.system_time += TIME_STEP
        if env.simulator.system_time >= toggle_time and env.simulator.system_time < toggle_time + toggle_duration:
            env.simulator.theta = toggle_theta
            if env.simulator.system_time == toggle_time:
                print("Toggle theta to", env.simulator.theta)
        else:
            env.simulator.theta = origin_theta
            if env.simulator.system_time == toggle_time + toggle_duration:
                print("Toggle theta back to", env.simulator.theta)
        env.simulator.run_cycle()
        
    rejection_rates = env.simulator.past_rejections
    anticipatory_costs = env.simulator.past_costs
    idle_veh_num_list = env.simulator.idle_veh_num_list
    users_costs = env.simulator.past_users_costs
    operators_costs = env.simulator.past_operators_costs
    return rejection_rates, anticipatory_costs, idle_veh_num_list, users_costs, operators_costs

def warm_up(env, duration):
    for _ in tqdm(range(int(duration/TIME_STEP)), desc=f"Warm-Up"):
        env.simulator.system_time += TIME_STEP
        env.simulator.run_cycle()
    return env

def policy_run(env, policy, sim_duration, fixed_normalizer):
    state, reward, done, _  = env.step(action = 0.0)
    state = fixed_normalizer(state)
    edge_index = env.graph_to_data(env.simulator.get_simulator_network_by_areas())
    auto_encoder = policy.gnn_auto_encoder.to(device)
    encoded_state, decoded_state = auto_encoder(state, edge_index, 1)

    for _ in tqdm(range(int(sim_duration/(TIME_STEP*5))), desc="Policy Run"):
        action = policy.select_action(encoded_state)
        next_state, reward, done, _ = env.step(action)
        next_state = fixed_normalizer(next_state)
        next_encoded_state, _ = auto_encoder(next_state, edge_index, 1)
        encoded_state = next_encoded_state

    rejection_rates = env.simulator.past_rejections
    anticipatory_costs = env.simulator.past_costs
    idle_veh_num_list = env.simulator.idle_veh_num_list
    users_costs = env.simulator.past_users_costs
    operators_costs = env.simulator.past_operators_costs
    theta_list = env.simulator.past_thetas
    
    return rejection_rates, anticipatory_costs, operators_costs, users_costs, idle_veh_num_list, theta_list

def non_policy_run(env, sim_duration, initial_theta):
    env.simulator.theta = initial_theta

    for _ in tqdm(range(int(sim_duration/TIME_STEP)), desc='Non-Policy Run'):
        env.simulator.system_time += TIME_STEP
        env.simulator.run_cycle()
        
    rejection_rates = env.simulator.past_rejections
    anticipatory_costs = env.simulator.past_costs
    idle_veh_num_list = env.simulator.idle_veh_num_list
    users_costs = env.simulator.past_users_costs
    operators_costs = env.simulator.past_operators_costs

    return rejection_rates, anticipatory_costs, operators_costs, users_costs, idle_veh_num_list

def toggle_theta_test(env, sim_duration, fixed_normalizer, toggle_theta, toggle_time, toggle_duration):
    state, reward, done, _  = env.step(action = -1)
    state = fixed_normalizer(state)


    for _ in tqdm(range(int(sim_duration/(TIME_STEP*5))), desc=f"Toggle Theta Test"):
        # print(env.simulator.theta)
        if env.simulator.system_time >= toggle_time and env.simulator.system_time < toggle_time + toggle_duration:
            action = toggle_theta
        else:
            action = -1
        next_state, reward, done, _ = env.step(action)
        if env.simulator.system_time == toggle_time + TIME_STEP*5 or env.simulator.system_time == toggle_time + toggle_duration + TIME_STEP*5:
            print(f"toggled theta: {env.simulator.theta}")
        
    rejection_rates = env.past_rejections
    anticipatory_costs = env.past_anticipatory_costs
    idle_veh_num_list = env.simulator.idle_veh_num_list
    return rejection_rates, anticipatory_costs, idle_veh_num_list