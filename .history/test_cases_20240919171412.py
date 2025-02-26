import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import pickle
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

# # Load Policy
# policy = TD3.TD3(**kwargs)
# policy.load(f"saved_models/{model_name}")

def toggle_test():
        # Main TEST CASE
    sim_duration = 3600 * 7  # 7 hours
    warm_up_duration = 3600 * 1 # 1 hour
    theta_duration = 1800 # 30 mins 
    initial_theta = 0.0


    toggle_periods_1 = [
        # all time zero theta
        {'start': 0, 'end': sim_duration, 'theta': 0.0},
    ]
    toggle_periods_2 = [
        {'start': warm_up_duration + 0, 'end': warm_up_duration + 0 + theta_duration, 'theta': 15.0},
        {'start': warm_up_duration + 7200, 'end': warm_up_duration + 7200 + theta_duration, 'theta': 30.0},
        {'start': warm_up_duration + 14400, 'end': warm_up_duration + 14400 + theta_duration, 'theta': 45.0},
        {'start': warm_up_duration + 21600, 'end': warm_up_duration + 21600 + theta_duration, 'theta': 60.0},
    ]

    toggle_periods_3 = [
        {'start': warm_up_duration + 0, 'end': warm_up_duration + 0 + theta_duration, 'theta': 15.0},
        # {'start': warm_up_duration + 7200, 'end': warm_up_duration + 7200 + theta_duration, 'theta': 30.0},
        {'start': warm_up_duration + 14400, 'end': warm_up_duration + 14400 + theta_duration, 'theta': 45.0},
        # {'start': warm_up_duration + 21600, 'end': warm_up_duration + 21600 + theta_duration, 'theta': 60.0},
    ]

    toggle_periods_4 = [
        {'start': warm_up_duration + 0, 'end': warm_up_duration + 0 + theta_duration, 'theta': 15.0},
        # {'start': warm_up_duration + 7200, 'end': warm_up_duration + 7200 + theta_duration, 'theta': 30.0},
        # {'start': warm_up_duration + 14400, 'end': warm_up_duration + 14400 + theta_duration, 'theta': 45.0},
        {'start': warm_up_duration + 21600, 'end': warm_up_duration + 21600 + theta_duration, 'theta': 60.0},
    ]

    # toggle_periods_list = [toggle_periods_1, toggle_periods_2, toggle_periods_3, toggle_periods_4]
    toggle_periods_list = [toggle_periods_1, toggle_periods_2]

    print('test')

    results = Parallel(n_jobs=2)(delayed(run_test_multi_toggle)(toggle_periods, initial_theta, sim_duration, warm_up_duration) 
                                for toggle_periods in toggle_periods_list)
    return results

class FixedNormalizer(object):
    def __init__(self):
        path = 'mean_std_data_100800.npz'
        assert os.path.exists(path), "mean_std_data_100800.npz not found"
        data = np.load(path)
        self.mean = torch.tensor(data['mean'], dtype=torch.float32).to(device)
        self.std = torch.tensor(data['std'], dtype=torch.float32).to(device)
        self.std[self.std == 0] = 1.0 # avoid division by zero

    def __call__(self, x):
        return (x - self.mean) / (self.std)
    
def run_test_multi_toggle(toggle_periods, initial_theta, sim_duration, warm_up_duration):
    test_env = ManhattanTrafficEnv()
    test_env.uniform_reset()
    test_env.simulator.theta = initial_theta
    env = warm_up(test_env, warm_up_duration)
    return toggle_theta_test_non_policy_multi_periods(env, toggle_periods, sim_duration, initial_theta)

def toggle_theta_test_non_policy_multi_periods(env, toggle_periods, sim_duration, initial_theta):
    env.simulator.theta = initial_theta
    # rejection_rates = []
    # costs = []
    # operators_costs = []
    # users_costs = []
    # idle_veh_num_list = []

    for _ in tqdm(range(int(sim_duration/TIME_STEP)), desc='Multi-Toggle Theta Test'):
        env.simulator.system_time += TIME_STEP
        current_theta = initial_theta

        for period in toggle_periods:
            if period['start'] <= env.simulator.system_time < period['end']:
                current_theta = period['theta']
                break

        env.simulator.theta = current_theta
        env.simulator.run_cycle()
        
    rejection_rates = env.simulator.past_rejections
    anticipatory_costs = env.simulator.past_costs
    idle_veh_num_list = env.simulator.idle_veh_num_list
    users_costs = env.simulator.past_users_costs
    operators_costs = env.simulator.past_operators_costs

    return rejection_rates, anticipatory_costs, operators_costs, users_costs, idle_veh_num_list

def plot_comparison_multi_toggle(data_1, data_2, sim_duration, warm_up_duration, toggle_periods_1, toggle_periods_2, type, n=12):
    time_scope = sim_duration + warm_up_duration
    if type == 'Reward':
        time_scope -= int(MEMORY_SIZE)*TIME_STEP
        n=1
    time_series = np.arange(0, time_scope, 15 * n)

    data_1_avg = [np.mean(data_1[i:i + n]) for i in range(0, len(data_1), n)]
    data_2_avg = [np.mean(data_2[i:i + n]) for i in range(0, len(data_2), n)]        
    diff_data_avg = [d2 - d1 for d2, d1 in zip(data_2_avg, data_1_avg)]

    def create_plot(data, color, label, title, toggle_periods):
        plt.figure(figsize=(12, 6))
        plt.plot(time_series, data, color=color)
        plt.xlabel('Time (s)')
        plt.ylabel(type)
        plt.title(title)
        plt.grid(True)
        if toggle_periods:  # Only add toggle period markers if there are any
            for period in toggle_periods:
                plt.axvline(x=period['start'], color='black', linestyle='--')
                plt.axvline(x=period['end'], color='black', linestyle='--')
                plt.axvspan(period['start'], period['end'], color='gray', alpha=0.3)
                plt.text(period['start'], plt.ylim()[0], f"{period['start']}s, θ={period['theta']}", ha='right', va='bottom', rotation=90)
                plt.text(period['end'], plt.ylim()[0], f"{period['end']}s", ha='left', va='bottom', rotation=90)
        
        # plt.legend()
        save_name = f'{title}_theta_' + '_'.join([f"{period['theta']}" for period in toggle_periods]) if toggle_periods else f'{title}_no_toggle'
        plt.savefig(f'pics/{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

    # create_plot(diff_data_avg, 'green', f'Difference in {type} (Policy - Base)', f'Difference in {type} (Policy - Base)', toggle_periods_2)
    create_plot(diff_data_avg, 'green', f'Difference in {type} (Toggle - Fixed)', f'Difference in {type} (Toggle - Fixed)', toggle_periods_2)

def plot_all_results_comparison(results, sim_duration, warm_up_duration, toggle_periods_list, n=12):
    if not toggle_periods_list:
        # If there are no toggle periods, just plot the results without comparisons
        metrics = ['Rejection Rate', 'Total Costs', "Operator's Costs", "User's Costs", 'Idle Vehicle Number']
        for i, metric in enumerate(metrics):
            plot_comparison_multi_toggle(
                results[0][i],
                results[1][i], 
                sim_duration,
                warm_up_duration,
                None,
                None,
                metric,
                n
            )
    else:
        base_toggle_periods = toggle_periods_list[0]
        toggle_periods_list = toggle_periods_list[1:]
        base_result = results[0]
        compare_results = results[1:]
        metrics = ['Rejection Rate', 'Total Costs', "Operator's Costs", "User's Costs", 'Idle Vehicle Number']

        for i, metric in enumerate(metrics):
            for j, toggle_periods in enumerate(toggle_periods_list):
                plot_comparison_multi_toggle(
                    base_result[i],
                    compare_results[j][i],
                    sim_duration,
                    warm_up_duration,
                    base_toggle_periods,
                    toggle_periods,
                    metric,
                    n
                )
def run_fixed_theta(initial_theta, sim_duration, warm_up_duration):
    test_env = ManhattanTrafficEnv()
    test_env.uniform_reset()
    test_env.simulator.theta = initial_theta
    env = warm_up(test_env, warm_up_duration)
    return non_policy_run(env, sim_duration, initial_theta)

def run_policy_theta(initial_theta, sim_duration, warm_up_duration):
    test_env = ManhattanTrafficEnv()
    fixed_normalizer = FixedNormalizer()
    test_env.uniform_reset()
    test_env.simulator.theta = initial_theta
    env = warm_up(test_env, warm_up_duration)
    # edge_index = test_env.graph_to_data(test_env.simulator.get_simulator_network_by_areas())
    
    return policy_run(env, policy, sim_duration, fixed_normalizer)

def decay_reward(reward_list):
    reward = 0
    for i in range(MEMORY_SIZE):
        reward += reward_list[i] * np.exp(-0.01 * i)
    reward /= MEMORY_SIZE
    return reward
def immediate_reward(rejection_rate, users_cost):
    current_reward = (REJ_THRESHOLD_REWARD - rejection_rate) * REJ_THRESHOLD_MULTIPLIER
    current_reward = max(-1.0, min(1.0, current_reward)) 
    extra_cost = users_cost - BASE_USER_COST
    anticipatory_reward = - extra_cost * ANTICIAPATORY_REWARD_MULTIPLIER
    return current_reward + anticipatory_reward

def calculate_rewards(rejection_rates, users_costs):
    current_rewards_list = []
    for rejection_rate, users_cost in zip(rejection_rates, users_costs):
        current_reward = immediate_reward(rejection_rate, users_cost)
        current_rewards_list.append(current_reward)
    # print(len(current_rewards_list))
    
    current_decay_reward_list = []
    while len(current_rewards_list) > MEMORY_SIZE:
        current_decay_reward = decay_reward(current_rewards_list)
        current_decay_reward_list.append(current_decay_reward)
        current_rewards_list.pop(0)
    # print(len(current_decay_reward_list))

    return current_decay_reward_list

def prune_results(results, sim_duration, warm_up_duration, initial_theta):
    # compare two results. prune the longer one
    base_result = results[0]
    policy_result = results[1]
    for i in range(len(base_result)):
        if len(base_result[i]) > len(policy_result[i]):
            base_result[i] = base_result[i][:len(policy_result[i])]
        else:
            policy_result[i] = policy_result[i][:len(base_result[i])]
    # add theta list to base result
    fixed_theta_list = [initial_theta] * len(policy_result[-1])
    base_result.append(fixed_theta_list)
    return [base_result, policy_result]

def check_result_length(results):
    for i in range(len(results)):
        for j in range(len(results[i])):
            print(len(results[i][j]))

def plot_both_multi_toggle(data_1, data_2, sim_duration, warm_up_duration, toggle_periods_1, toggle_periods_2, type, n=12):
    time_scope = sim_duration + warm_up_duration
    if type == 'Reward':
        time_scope -= int(MEMORY_SIZE)*TIME_STEP
        n=1
    time_series = np.arange(0, time_scope, 15 * n)

    data_1_avg = [np.mean(data_1[i:i + n]) for i in range(0, len(data_1), n)]
    data_2_avg = [np.mean(data_2[i:i + n]) for i in range(0, len(data_2), n)]        

    def create_plot(data_1, data_2, color_1, color_2, label_1, label_2, title, toggle_periods):
        plt.figure(figsize=(12, 6))
        plt.plot(time_series, data_1, color=color_1, label=label_1)
        plt.plot(time_series, data_2, color=color_2, label=label_2)
        plt.xlabel('Time (s)')
        plt.ylabel(type)
        plt.title(title)
        plt.grid(True)
        if toggle_periods:  # Only add toggle period markers if there are any
            for period in toggle_periods:
                plt.axvline(x=period['start'], color='black', linestyle='--')
                plt.axvline(x=period['end'], color='black', linestyle='--')
                plt.axvspan(period['start'], period['end'], color='gray', alpha=0.3)
                plt.text(period['start'], plt.ylim()[0], f"{period['start']}s, θ={period['theta']}", ha='right', va='bottom', rotation=90)
                plt.text(period['end'], plt.ylim()[0], f"{period['end']}s", ha='left', va='bottom', rotation=90)
        
        plt.legend()
        save_name = f'{title}_theta_' + '_'.join([f"{period['theta']}" for period in toggle_periods]) if toggle_periods else f'{title}_no_toggle'
        plt.savefig(f'pics/{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

    # create_plot(data_1_avg, data_2_avg, 'blue', 'red', 'Base', 'Policy', f'{type}', toggle_periods_2)
    create_plot(data_1_avg, data_2_avg, 'blue', 'red', 'Base', 'Fixed', f'{type}', toggle_periods_2)

def plot_all_results_both(results, sim_duration, warm_up_duration, toggle_periods_list, n=12):
    if not toggle_periods_list:
        # If there are no toggle periods, just plot the results without comparisons
        metrics = ['Rejection Rate', 'Total Costs', "Operator's Costs", "User's Costs", 'Idle Vehicle Number']
        for i, metric in enumerate(metrics):
            plot_both_multi_toggle(
                results[0][i],
                results[1][i], 
                sim_duration,
                warm_up_duration,
                None,
                None,
                metric,
                n
            )
    else:
        base_toggle_periods = toggle_periods_list[0]
        toggle_periods_list = toggle_periods_list[1:]
        base_result = results[0]
        compare_results = results[1:]
        metrics = ['Rejection Rate', 'Total Costs', "Operator's Costs", "User's Costs", 'Idle Vehicle Number']

        for i, metric in enumerate(metrics):
            for j, toggle_periods in enumerate(toggle_periods_list):
                plot_both_multi_toggle(
                    base_result[i],
                    compare_results[j][i],
                    sim_duration,
                    warm_up_duration,
                    base_toggle_periods,
                    toggle_periods,
                    metric,
                    n
                )
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

if __name__ == "__main__":
    sim_duration = 3600 * 7  # 7 hours
    warm_up_duration = 3600 * 1 # 1 hour
    theta_duration = 1800 # 30 mins 
    initial_theta = 0.0

    # toggle_periods_1 = [
    #     # all time zero theta
    #     {'start': 0, 'end': sim_duration, 'theta': 0.0},
    # ]
    # toggle_periods_2 = [
    #     {'start': warm_up_duration + 0, 'end': warm_up_duration + 0 + theta_duration, 'theta': 15.0},
    #     {'start': warm_up_duration + 7200, 'end': warm_up_duration + 7200 + theta_duration, 'theta': 30.0},
    #     {'start': warm_up_duration + 14400, 'end': warm_up_duration + 14400 + theta_duration, 'theta': 45.0},
    #     {'start': warm_up_duration + 21600, 'end': warm_up_duration + 21600 + theta_duration, 'theta': 60.0},
    # ]

    # toggle_periods_3 = [
    #     {'start': warm_up_duration + 0, 'end': warm_up_duration + 0 + theta_duration, 'theta': 15.0},
    #     # {'start': warm_up_duration + 7200, 'end': warm_up_duration + 7200 + theta_duration, 'theta': 30.0},
    #     {'start': warm_up_duration + 14400, 'end': warm_up_duration + 14400 + theta_duration, 'theta': 45.0},
    #     # {'start': warm_up_duration + 21600, 'end': warm_up_duration + 21600 + theta_duration, 'theta': 60.0},
    # ]

    # toggle_periods_4 = [
    #     {'start': warm_up_duration + 0, 'end': warm_up_duration + 0 + theta_duration, 'theta': 15.0},
    #     # {'start': warm_up_duration + 7200, 'end': warm_up_duration + 7200 + theta_duration, 'theta': 30.0},
    #     # {'start': warm_up_duration + 14400, 'end': warm_up_duration + 14400 + theta_duration, 'theta': 45.0},
    #     {'start': warm_up_duration + 21600, 'end': warm_up_duration + 21600 + theta_duration, 'theta': 60.0},
    # ]

    # # toggle_periods_list = [toggle_periods_1, toggle_periods_2, toggle_periods_3, toggle_periods_4]
    # toggle_periods_list = [toggle_periods_1, toggle_periods_2]

    
    # results = Parallel(n_jobs=2)(delayed(run_test_multi_toggle)(toggle_periods, initial_theta, sim_duration, warm_up_duration) 
    #                           for toggle_periods in toggle_periods_list)
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

    epoch = 36
    model_name = f'TD3_AUTOENCODER_{epoch}'

    # Load Policy
    policy = TD3.TD3(**kwargs)
    policy.load(f"saved_models/{model_name}")

    # Define parameters for both runs
    theta_method = 'Fixed'
    initial_theta = 30.0
    sim_duration = 3600 * 15  # 7 hours
    warm_up_duration = 3600 * 1  # 1 hour
    n=20

    
    # 使用pickle保存结果
    with open('simulation_results.pkl', 'wb') as f:
        pickle.dump(results, f)