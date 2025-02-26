import numpy as np
import torch
import gym
import argparse
import os
import torch.multiprocessing as mp

import src.RL.utils as utils
# import src.RL.TD3 as TD3
from src.RL.environment import ManhattanTrafficEnv
from src.simulator.config import *
from tqdm import tqdm
import math
import random   

def collect_data(rank, args, replay_queue):
    print("---------------------------------------")
    print(f"Process {rank} - Collecting Data")
    print("---------------------------------------")

    # 初始化环境和策略
    env = ManhattanTrafficEnv()

    state_dim = env.observation_space.shape[0]
    # state_dim = 63 * (32 + NUM_FEATURES)
    action_dim = env.action_space.shape[0]
   
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Warm up the agent before start training
    system_time = 0.0
    while system_time < WARM_UP_DURATION:
    # while False:
        warm_up_done = False
        warm_up_step = 0
        while not warm_up_done:
            warm_up_step += 1
            env.simulator.current_cycle_rej_rate = []  # reset the rejection rate to record the new cycle
            for step in tqdm(range(int(RL_STEP_LENGTH)), desc=f"WarmUP_{warm_up_step}"):  # 2.5mins, 10 steps
                env.simulator.system_time += TIME_STEP
                env.simulator.run_cycle()  # Run one cycle(15s)
                system_time = env.simulator.system_time
                warm_up_done = env.warm_up_step()

    state, reward, done, _  = env.step(action=0.0)

    for t in range(int(args.data_collection_step-1)):
        print(f"step: {t}")
        
        action = env.action_space.sample()

        # Perform action
        next_state, reward, done, _ = env.step(action)  # shape: [63,7]
        done_bool = float(done)

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        if done:
            print(f"Process {rank} - Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    # 将当前进程的 replay buffer 数据传递到主进程
    replay_queue.put(replay_buffer.get_all_data())
    # return replay_queue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="ManhattanTrafficEnv")     # Doesn't matter. Overriden by ManhattanTrafficEnv
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    # parser.add_argument("--start_timesteps", default=100, type=int) # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--data_collection_step", default=36, type=int)   # how many steps to collect data each run
    parser.add_argument("--expl_noise", default=0.2, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=32, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)    # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--num_processes", default=96, type=int)     # Number of processes to use 
    parser.add_argument("--training_repeat", default=1, type=int)   # Number of times to train the policy
    parser.add_argument("--cycles", default=1000, type=int)           # Number of cycles to run the data_collection and training
    args = parser.parse_args()
    
    print("---------------------------------------")
    print(f"CPU Cores: {args.num_processes}")
    print("---------------------------------------")

    # 初始化策略
    env = ManhattanTrafficEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(1e6)) 

    # 使用 torch.multiprocessing 并行训练任务
    manager = mp.Manager()
    replay_queue = manager.Queue()
    
    # data_collection and training
    for cycle in range(args.cycles):
        # policy_params = torch.load('policy_checkpoint.pth')
        # multiprocessing
        processes = []
        for i in range(args.num_processes):
            p = mp.Process(target=collect_data, args=(i, args, replay_queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # 从 replay_queue 中获取所有进程的数据并合并到一个 replay buffer 中
        while not replay_queue.empty():
            data = replay_queue.get()
            replay_buffer.merge(data)

        print(f"Replay buffer size: {replay_buffer.size}")
        save_freq = args.num_processes * (args.data_collection_step - 1) * 10
        if replay_buffer.size%save_freq == 0: 
            mean, std = replay_buffer.get_mean_std()
            np.savez('mean_std_data.npz', mean=mean, std=std)
            bufffer_name = f'replay_buffer_data_{replay_buffer.size}.npz'
            replay_buffer.save(bufffer_name)

if __name__ == '__main__':
    main()
