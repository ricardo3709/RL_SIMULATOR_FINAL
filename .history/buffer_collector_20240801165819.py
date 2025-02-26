import numpy as np
import torch
import gym
import argparse
import os
import torch.multiprocessing as mp

import src.RL.utils as utils
import src.RL.TD3 as TD3
from src.RL.environment import ManhattanTrafficEnv
from src.simulator.config import *
from tqdm import tqdm
import math
import random   
# Check device availability: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    # print("Using CUDA device")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    # print("Using MPS device")
else:
    device = torch.device("cpu")

if device == torch.device("cuda"):
        mp.set_start_method('spawn', force=True)
num_gpus = torch.cuda.device_count()

# 定义选择策略动作的概率
policy_prob = 0.9

def eval_policy(policy, eval_env, seed, eval_episodes=1 ):
    print("---------------------------------------")
    print("Evaluation")
    print("---------------------------------------")
    eval_env.seed(seed + 100)
    avg_reward = 0.

    actor_losses = []
    critic_losses = []
    gnn_encoder_losses = []
    rewards = []
    actions = []

    network = eval_env.simulator.get_simulator_network_by_areas()
    edge_index = eval_env.graph_to_data(network)

    # running_state = ZFilter(((63,NUM_FEATURES)), clip=5.0)
    fixed_normalizer = FixedNormalizer()

    # Warm up the agent before start evaluation
    system_time = 0.0
    while system_time < WARM_UP_DURATION:
        warm_up_done = False
        warm_up_step = 0
        while not warm_up_done:
            warm_up_step += 1
            eval_env.simulator.current_cycle_rej_rate = []  # reset the rejection rate to record the new cycle
            for step in tqdm(range(int(RL_STEP_LENGTH)), desc=f"WarmUP_{warm_up_step}"):  # 2.5mins, 10 steps
                eval_env.simulator.system_time += TIME_STEP
                eval_env.simulator.run_cycle()  # Run one cycle(15s)
                system_time = eval_env.simulator.system_time
                warm_up_done = eval_env.warm_up_step()

    state, reward, done, _  = eval_env.step(action = 0.0)
    state = fixed_normalizer(state)
    for _ in range(eval_episodes):
        # state, done = eval_env.reset(), False
        # state = running_state(state.detach().cpu()).to(device)
        
        for _ in range(10):
        # while not done:
            encoder = policy.gnn_encoder
            encoded_state = encoder(state, edge_index, 1)
            # action = policy.select_action(state.cpu().detach().numpy())

            # action = policy.select_action(encoded_state.cpu().detach().numpy())
            action_with_grad = policy.select_action(encoded_state)
            
            action = float(action_with_grad)
            
            state, reward, done, _ = eval_env.step(action)

            # state = running_state(state.detach().cpu()).to(device)
            state = fixed_normalizer(state)
            # state = state.to(device)

            actor_loss, critic_loss, gnn_encoder_loss = policy.get_loss()
            # if type(actor_loss) == int and type(critic_loss) == int:
            #     actor_loss = torch.tensor(actor_loss)
            #     critic_loss = torch.tensor(critic_loss)
            # actor_losses.append(actor_loss.cpu().detach().numpy())
            # critic_losses.append(critic_loss.cpu().detach().numpy())
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            gnn_encoder_losses.append(gnn_encoder_loss)
            rewards.append(reward)
            actions.append(action)
            # avg_reward += reward
            # done = True

    avg_reward = np.mean(rewards)
    avg_actor_loss = np.mean(actor_losses)
    avg_critic_loss = np.mean(critic_losses)
    avg_gnn_encoder_loss = np.mean(gnn_encoder_losses)
    avg_action = np.mean(actions)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes")
    print(f"Average Actor Loss: {avg_actor_loss:.3f}")
    print(f"Average Critic Loss: {avg_critic_loss:.3f}")
    print(f"Average GNN Encoder Loss: {avg_gnn_encoder_loss:.3f}")
    print(f"Average Avtion: {avg_action:.3f}")
    print(f"Average Reward: {avg_reward:.3f}")
    print("---------------------------------------")

    return avg_reward, avg_actor_loss, avg_critic_loss, avg_gnn_encoder_loss, avg_action

def collect_data(rank, args, seed_offset, replay_queue, policy_flag, cycle):
    print("---------------------------------------")
    print(f"Process {rank} - Collecting Data")
    print("---------------------------------------")
    if device == torch.device("cuda"):
        gpu_id = rank%args.num_gpus
        torch.cuda.set_device(gpu_id)
        print(f"Rank: {rank}, running on GPU: {gpu_id}")

    # 初始化环境和策略
    env = ManhattanTrafficEnv()
    env.seed(args.seed + seed_offset)
    env.action_space.seed(args.seed + seed_offset)
    torch.manual_seed(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)

    state_dim = env.observation_space.shape[0]
    # state_dim = 63 * (32 + NUM_FEATURES)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
    }

    policy = TD3.TD3(**kwargs)
    policy_params_1 = torch.load('policy_checkpoint.pth')
    policy.load_dict(policy_params_1)

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    fixed_normalizer = FixedNormalizer()
    # running_state = ZFilter(((63,NUM_FEATURES)), clip=5.0)

    state, done = env.reset(), False
    state = fixed_normalizer(state)
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
    state = fixed_normalizer(state)

    for t in range(int(args.data_collection_step-1)):
        # episode_timesteps += 1
        print(f"step: {t}")
        # Select action randomly or according to policy
        # if t < args.start_timesteps:
        if not policy_flag:
            action = env.action_space.sample()
            # action_with_grad = torch.tensor(action).to(device)
        else:
            # 随机选择动作
            random_idx = random.random() * cycle 
            if random_idx < 1 - policy_prob:
                action = env.action_space.sample()
            else:
                network_temp = env.simulator.get_simulator_network_by_areas()
                edge_index = env.graph_to_data(network_temp)
                encoder = policy.gnn_encoder.to(device)
                encoded_state = encoder(state, edge_index, 1)
                action_with_grad = policy.select_action(encoded_state)
                action = (
                    float(action_with_grad)
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)  # shape: [63,7]
        # done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        done_bool = float(done)

        # next_state = running_state(next_state.detach().cpu()).to(device)
        next_state = fixed_normalizer(next_state)

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


# def train(args, policy_params, replay_queue, cycle):
# def train(args, policy_params, replay_buffer, cycle):
def train(args, replay_buffer, cycle):
    print("---------------------------------------")
    print(f"Training Start")
    print("---------------------------------------")
    if device == torch.device("cuda"):
        torch.cuda.set_device(0)

    # 初始化策略
    env = ManhattanTrafficEnv()
    # state_dim = env.observation_space.shape[0]
    state_dim = 63 * (32 + NUM_FEATURES)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
    }

    policy = TD3.TD3(**kwargs)
    # encoder = TD3.GNN_Encoder()
    policy_params = torch.load('policy_checkpoint.pth')
    policy.load_dict(policy_params)
    policy = policy.to(device)

    # replay_max_size = args.num_processes * args.data_collection_step
    # replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=replay_max_size)

    network = env.simulator.get_simulator_network_by_areas()
    edge_index = env.graph_to_data(network)

    # # 从 replay_queue 中获取所有进程的数据并合并到一个 replay buffer 中
    # while not replay_queue.empty():
    #     data = replay_queue.get()
    #     replay_buffer.merge(data)
    batch_size, training_repeat = get_batch_size_and_update_times(args, replay_buffer)
    print(f"Batch Size: {batch_size} | Training Repeat: {training_repeat}")
    for _ in range(training_repeat):
        policy.train(replay_buffer, batch_size, edge_index)

    avg_reward, avg_actor_loss, avg_critic_loss, avg_gnn_encoder_loss, avg_action = eval_policy(policy, env, args.seed)
    with open('training_log.txt', 'a') as log_file:
        log_file.write(f"Epoch:{cycle} | Actor_Loss:{avg_actor_loss:.3f} | Critic_Loss:{avg_critic_loss:.3f} | GNN_Encoder_Loss:{avg_gnn_encoder_loss:.3f} | Avg_Reward:{avg_reward:.3f} | Avg_Action:{avg_action:.3f}\n")
    
    policy = policy.to('cpu')
    policy_params = policy.state_dict()
    torch.save(policy_params, 'policy_checkpoint.pth')
    # return policy_params

def get_batch_size_and_update_times(args, replay_buffer):
    replay_len = replay_buffer.size
    basic_batch_size = args.batch_size
    basic_update_times = args.training_repeat

    # 计算 k 的值
    k = math.sqrt(0.5 * replay_len / (basic_batch_size * basic_update_times))

    # 计算初步的 batch_size
    batch_size_temp = int(k * basic_batch_size)

    # 调整 batch_size 为 2 的 n 次方
    batch_size = 2 ** (math.floor(math.log2(batch_size_temp + 1)) or 1)  # 确保至少为 2

    # 重新计算 update_times 以满足条件 batch_size * update_times = 0.5 * replay_len
    update_times = max(1, int(0.5 * replay_len / batch_size))

    # batch_size < 256, update_times<10
    batch_size = min(batch_size, 256)
    update_times = min(update_times, 10)
    
    return batch_size, update_times

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
    parser.add_argument("--num_processes", default=64, type=int)     # Number of processes to use 
    parser.add_argument("--training_repeat", default=1, type=int)   # Number of times to train the policy
    parser.add_argument("--cycles", default=1000, type=int)           # Number of cycles to run the data_collection and training
    parser.add_argument("--num_gpus", default=num_gpus, type=int) #available gpu numbers
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, CPU Cores: {args.num_processes}")
    print("---------------------------------------")
    
    with open('training_log.txt', 'w') as log_file:
        log_file.write("Training Log\n")
    
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./saved_models"):
        os.makedirs("./saved_models")

    # 初始化策略
    env = ManhattanTrafficEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=1e6) 

    # 使用 torch.multiprocessing 并行训练任务
    manager = mp.Manager()
    replay_queue = manager.Queue()
    return_dict = manager.dict()
    
    # data_collection and training
    for cycle in range(args.cycles):
        # policy_params = torch.load('policy_checkpoint.pth')
        # multiprocessing
        processes = []
        seed_offsets = [seed_idx for seed_idx in range(args.num_processes)]
        policy_flag = True if cycle > 0 else False
        for i in range(args.num_processes):
            p = mp.Process(target=collect_data, args=(i, args, seed_offsets[i], replay_queue, policy_flag, cycle))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # 从 replay_queue 中获取所有进程的数据并合并到一个 replay buffer 中
        while not replay_queue.empty():
            data = replay_queue.get()
            replay_buffer.merge(data)

        print(f"Replay buffer size: {replay_buffer.size}")
        if replay_buffer.size > 0:
            mean, std = replay_buffer.get_mean_std()
            np.savez('mean_std_data.npz', mean=mean, std=std)

class FixedNormalizer(object):
    def __init__(self):
        path = 'imac_200_mean_std_data.npz'
        assert os.path.exists(path), "imac_200_mean_std_data.npz not found"
        data = np.load(path)
        self.mean = torch.tensor(data['mean'], dtype=torch.float32).to(device)
        self.std = torch.tensor(data['std'], dtype=torch.float32).to(device)
        self.std[self.std == 0] = 1.0 # avoid division by zero

    def __call__(self, x):
        return (x - self.mean) / (self.std)


if __name__ == '__main__':
    main()
