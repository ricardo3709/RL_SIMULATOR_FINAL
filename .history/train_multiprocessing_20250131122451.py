import numpy as np
import torch
import gym
import argparse
import os
import torch.multiprocessing as mp
import psutil
import copy

import pickle
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

def eval_policy(policy, eval_env, seed, eval_episodes=1):
    print("---------------------------------------")
    print("Evaluation")
    print("---------------------------------------")
    avg_reward = 0

    actor_losses = []
    critic_losses = []
    gnn_encoder_losses = []
    rewards = []
    delayed_reward_list = []
    actions = []

    network = eval_env.simulator.get_simulator_network_by_areas()
    edge_index = eval_env.graph_to_data(network)

    # running_state = ZFilter(((NUM_AREA,NUM_FEATURES)), clip=5.0)
    fixed_normalizer = FixedNormalizer()

    torch.manual_seed(seed + 200)
    np.random.seed(seed + 200)
    random.seed(seed + 200)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + 200)
        torch.cuda.manual_seed_all(seed + 200)

    
    # auto_encoder = policy.gnn_auto_encoder.to(device)
    gnn_encoder = policy.gnn_encoder.to(device)

    for current_ep in range(eval_episodes):
        eval_env.simulator.uniform_reset_simulator()
        eval_env.simulator.theta = 20.0
        # Warm up the agent before start evaluation
        system_time = 0.0
        # WARM_UP_DURATION = 0
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

        # origin_theta_action = eval_env.simulator.theta / (MAX_THETA/2) -1
        state, reward, done, _  = eval_env.step(action = 0)
        state = fixed_normalizer(state)
        
        # EVAL_STEPS = 1
        for step in range(int(EVAL_STEPS)): 
        # while not done:
            # get current theta and rej rate for state
            rej_rate = np.mean(eval_env.simulator.current_cycle_rej_rate)
            state_theta = eval_env.simulator.theta/MAX_THETA # normalize theta

            # encoded_state, decoded_state, _ = auto_encoder(state, edge_index, 1, rej_rate, state_theta)
            encoded_state = gnn_encoder(state, edge_index, 1, rej_rate, state_theta)
            
            action_with_grad = policy.select_action(encoded_state)
            
            action = float(action_with_grad)
            print("current action: ", action)
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
            if len(rewards) > MEMORY_SIZE:
                delayed_reward = decay_reward(rewards)
                delayed_reward_list.append(delayed_reward)
                rewards.pop(0)

    # avg_reward = np.mean(delayed_reward_list)
    tot_reward = sum(delayed_reward_list)
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
    print(f"Total Reward: {tot_reward:.3f}")
    print("---------------------------------------")

    return tot_reward, avg_actor_loss, avg_critic_loss, avg_gnn_encoder_loss, avg_action

# def collect_data(rank, args, seed_offset, replay_queue, policy_flag, cycle, env, return_env_queue):
def collect_data(rank, args, seed_offset, replay_queue, policy_flag, cycle):

    if args.num_processes == 8:
        P_CORE_INDICES = [0,2,4,6,8,10,12,14]  # Based on the provided mapping
        try:
            process = psutil.Process()
            core_id = P_CORE_INDICES[rank]  # Use rank to map to P-core indices
            process.cpu_affinity([core_id])
        except Exception as e:
            print(f"Cannot set CPU affinity: {e}")

    print("---------------------------------------")
    print(f"Process {rank} - Collecting Data")
    print("---------------------------------------")
    if device == torch.device("cuda"):
        gpu_id = rank%args.num_gpus
        torch.cuda.set_device(gpu_id)
        print(f"Rank: {rank}, running on GPU: {gpu_id}")

    # save_env_name = f"envs/env_{rank}.pkl"
    # with open(save_env_name, 'rb') as f:
    #     env = pickle.load(f)
        # 加载环境时使用try-except保护
    try:
        save_env_name = f"envs/env_{rank}.pkl"
        with open(save_env_name, 'rb') as f:
            env = pickle.load(f)
    except Exception as e:
        print(f"Process {rank} - Error loading environment: {e}")
        return

    # if env.simulator.system_time >= 8 * 3600: # 8 hrs of simulation finished
    #     env.simulator.random_reset_simulator() 

    process_seed = args.seed + seed_offset + rank * 1000
    torch.manual_seed(process_seed)
    np.random.seed(process_seed)
    random.seed(process_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(process_seed)
        torch.cuda.manual_seed_all(process_seed)

    env.seed(process_seed)
    env.action_space.seed(process_seed)

    state_dim = NUM_AREA*NUM_FEATURES
    action_dim = 1
    # max_action = float(env.action_space.high[0])
    max_action = args.max_action

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

    pre_epochs = args.load_model.split('_')[-1] if args.load_model != "" else 0
    if pre_epochs == 'BASE':
        pre_epochs = 0
    else:
        pre_epochs = int(pre_epochs)+1
    current_cycle = cycle + int(pre_epochs)

    policy = TD3.TD3(**kwargs)
    policy_params = torch.load('policy_checkpoint.pth', weights_only=True)
    policy.load_dict(policy_params)

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, args.data_collection_step)

    fixed_normalizer = FixedNormalizer()
    # running_state = ZFilter(((NUM_AREA,NUM_FEATURES)), clip=5.0)

    # state, done = env.reset(), False

    state = env.get_state_tensor()
    state = fixed_normalizer(state)
    episode_reward = 0

    network_temp = env.simulator.get_simulator_network_by_areas()   
    edge_index = env.graph_to_data(network_temp)

    # store state, next_state, action, reward, done
    state_list = []
    next_state_list = []
    action_list = []
    reward_list = []
    done_list = []
    rej_rate_list = []
    state_theta_list = []

    encoder = policy.gnn_encoder.to(device)
    # auto_encoder = policy.gnn_auto_encoder.to(device)

    # only warm up the env at the very beginning
    system_time = env.simulator.system_time
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

    # origin_theta_action = env.simulator.theta / (MAX_THETA/2) -1
    state, reward, done, _  = env.step(action = 0) 
    state = fixed_normalizer(state)

    for t in range(int(args.data_collection_step)):
        print(f"step: {t}, epoch: {current_cycle}")

        # get next_state theta and rej rate
        rej_rate = np.mean(env.simulator.current_cycle_rej_rate)
        state_theta = env.simulator.theta/MAX_THETA # normalize theta

        policy_prob = min(0.3 + 0.9 * (current_cycle / args.cycles), 0.9)
        # Select action randomly or according to policy
        # if t < args.start_timesteps:
        if not policy_flag:
            action = env.action_space.sample()
            print(f"Random Action: {action}")
            # action_with_grad = torch.tensor(action).to(device)
        else:
            # 随机选择动作
            # random_idx = random.random() * cycle 
            if random.random() < policy_prob:
                #policy action

                encoded_state = encoder(state, edge_index, 1, rej_rate, state_theta)
                # encoded_state, decoded_state, _ = auto_encoder(state, edge_index, 1, rej_rate, state_theta)

                policy_prob, current_noise = get_exploration_params(current_cycle, args.cycles)

                # current_noise = args.expl_noise * NOISE_DECAY_RATE ** cycle
                # current_noise = max(current_noise, 0.01)

                action_with_grad = policy.select_action(encoded_state)
                noise = torch.normal(0, max_action * current_noise, size=action_with_grad.shape).to(device)
                action = action_with_grad + noise
                action = action.clamp(-max_action, max_action)
                action = action.detach().cpu().numpy()
                print(f"REAL Policy Action: {action_with_grad}")
                print(f"Action with Noise: {action}")
            else:
                #random action
                action = env.action_space.sample()
                print(f"Random Action: {action}")
                
        # Perform action
        next_state, reward, _, _ = env.step(action)  # shape: [NUM_AREA,NUM_FEATURES]
        if t == args.data_collection_step-1: # last step
            done = True
        else:
            done = False
        done_bool = float(done)

        # next_state = running_state(next_state.detach().cpu()).to(device)
        next_state = fixed_normalizer(next_state)

        state_list.append(state)
        next_state_list.append(next_state)
        action_list.append(action)
        reward_list.append(reward)
        done_list.append(done_bool)
        rej_rate_list.append(rej_rate)
        state_theta_list.append(state_theta)
        # Store data in replay buffer
        # replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # if length of state_list > MEMORY_SIZE, calculate the reward of the first state in the list, and add it to the replay buffer
        if len(state_list) > MEMORY_SIZE:
            reward = decay_reward(reward_list)
            replay_buffer.add(state_list[0], action_list[0], next_state_list[0], reward, done_list[0], rej_rate_list[0], state_theta_list[0])
            state_list.pop(0)
            action_list.pop(0)
            next_state_list.pop(0)
            reward_list.pop(0)
            done_list.pop(0)
            rej_rate_list.pop(0)
            state_theta_list.pop(0)

        if done:
            print(f"Process {rank} - Finished")
            done = False
            state = env.reset()

    # 将当前进程的 replay buffer 数据传递到主进程
    replay_queue.put(replay_buffer.get_all_data())

    # return replay_queue
    # return_env_queue.put((rank, env))

    with open(save_env_name, 'wb') as f:
        pickle.dump(env, f)

def decay_reward(reward_list_input):
    reward_list = copy.deepcopy(reward_list_input)
    reward = 0
    
    # 计算权重
    weights = [1 - i/MEMORY_SIZE for i in range(MEMORY_SIZE)]
    # 计算归一化因子
    normalizer = sum(weights)
    # 标准化权重使其和为1
    normalized_weights = [w/normalizer for w in range(MEMORY_SIZE)]
    
    # 使用标准化后的权重计算reward
    for i in range(MEMORY_SIZE):
        reward += reward_list[i] * normalized_weights[i]
    
    # scaled_reward = reward * REWARD_MULTIPLIER 
    scaled_reward = reward / MAX_REWARD
    delayed_reward = np.clip(scaled_reward, -1.0, 1.0)
    
    print("Original Delayed Reward: ", reward)
    print("Output Delayed Reward: ", delayed_reward)
    
    # return reward #should be scaled_reward
    return delayed_reward

    # exponential decay
    # reward_list = copy.deepcopy(reward_list_input)
    # reward = 0
    # gamma = 0.7  # 衰减率（可调整）

    # # 计算归一化因子，确保权重和为1
    # normalizer = sum([gamma**i for i in range(MEMORY_SIZE)])

    # for i in range(MEMORY_SIZE):
    #     reward += reward_list[i] * (gamma**i)
    # reward = reward / normalizer  # 归一化

    # scaled_reward = reward * REWARD_MULTIPLIER
    # delayed_reward = np.clip(scaled_reward, -1.0, 1.0)

    # print("Clipped DELAYED REWARD: ", delayed_reward)
    # # print("Original DELAYED REWARD: ", reward)
    # # print("Scaled DELAYED REWARD: ", scaled_reward)
    # return delayed_reward

def train(args, replay_buffer, cycle):
    print("---------------------------------------")
    print(f"Training Start")
    print("---------------------------------------")
    if device == torch.device("cuda"):
        torch.cuda.set_device(0)

    # 初始化策略
    env = ManhattanTrafficEnv()
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    # state_dim = env.observation_space.shape[0]
    state_dim = NUM_AREA * 9
    action_dim = env.action_space.shape[0]
    max_action = args.max_action

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
    policy_params = torch.load('policy_checkpoint.pth', weights_only=True)
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
        policy.save_training_logs("./logs/training_logs.json")
    
    pre_epochs = args.load_model.split('_')[-1] if args.load_model != "" else 0
    if pre_epochs == 'BASE':
        pre_epochs = 0
    else:
        pre_epochs = int(pre_epochs)+1
    current_cycle = cycle + int(pre_epochs)
    if current_cycle % args.eval_freq == 0:
        tot_reward, avg_actor_loss, avg_critic_loss, avg_gnn_encoder_loss, avg_action = eval_policy(policy, env, args.seed) 
        with open('training_log.txt', 'a') as log_file:
            log_file.write(f"Epoch:{cycle+int(pre_epochs)} | Actor_Loss:{avg_actor_loss:.3f} | Critic_Loss:{avg_critic_loss:.3f} | GNN_Encoder_Loss:{avg_gnn_encoder_loss:.3f} | Total_Reward:{tot_reward:.3f} | Avg_Action:{avg_action:.3f}\n")
        
    policy = policy.to('cpu')
    
    if current_cycle > 0 and current_cycle%args.save_freq == 0:
        file_name = f"{args.policy}_{current_cycle}"
        policy = policy.to(device)
        policy.save(f"./saved_models/{file_name}")

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
    parser.add_argument("--policy", default="TD3_AUTOENCODER")       # Policy name 
    parser.add_argument("--env", default="ManhattanTrafficEnv")     # Doesn't matter. Overriden by ManhattanTrafficEnv
    parser.add_argument("--seed", default=42, type=int)              # Sets Gym, PyTorch and Numpy seeds
    # parser.add_argument("--start_timesteps", default=100, type=int) # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5, type=int)       # How often (epochs) we evaluate
    parser.add_argument("--save_freq", default=5, type=int)       # How often (epochs) we save
    parser.add_argument("--data_collection_step", default=48*4, type=int)   # how many steps to collect data each run 48*8 (8hours)
    parser.add_argument("--expl_noise", default=0.3, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=32, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.95, type=float)     # Discount factor, since reward is already delayed, we set as 0.95 to only consider a little bit of future reward
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update 0.1
    parser.add_argument("--noise_clip", default=0.2, type=float)    # Range to clip target policy noise 0.15
    parser.add_argument("--policy_freq", default=1, type=int)       # Frequency of delayed policy updates 2
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="TD3_AUTOENCODER_30") # Model load file name, "" doesn't load, "default" uses file_name TD3_AUTOENCODER_
    parser.add_argument("--num_processes", default=16, type=int)     # Number of processes to use 
    parser.add_argument("--training_repeat", default=1, type=int)   # Number of times to train the policy
    parser.add_argument("--cycles", default=200, type=int)           # Number of cycles to run the data_collection and training
    parser.add_argument("--num_gpus", default=num_gpus, type=int) #available gpu numbers
    parser.add_argument("--max_action", default=1.0, type=float)    # Max action value
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

    # delete checkpoint file
    if os.path.exists('policy_checkpoint.pth'):
        os.remove('policy_checkpoint.pth')
    # delete env files
    if os.path.exists('envs'):
        for file in os.listdir('envs'):
            os.remove(f'envs/{file}')

    # delete training log file
    if os.path.exists('logs'):
        # keep log file if load model
        if args.load_model == "":
            for file in os.listdir('logs'):
                os.remove(f'logs/{file}')


    # 设置全局随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    state_dim = NUM_AREA*NUM_FEATURES
    action_dim = 1
    max_action = args.max_action

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if True:
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
        policy_params = policy.state_dict()
        policy = policy.to('cpu')
        torch.save(policy_params, 'policy_checkpoint.pth')
        # encoder = TD3.GNN_Encoder()

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./saved_models/{policy_file}")
        policy_params = policy.state_dict()
        policy = policy.to('cpu')
        # re-initialize the optimizer
        policy.actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=0.001)
        torch.save(policy_params, 'policy_checkpoint.pth')

    replay_buffer_max_size = 50 * args.num_processes * args.data_collection_step
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(replay_buffer_max_size)) 

    # Evaluate untrained policy
    evaluations = []
    # evaluations = [eval_policy(policy, env, args.seed)]
    # result_line = f"Reward: {evaluations[-1][0]:.3f}, Actor Loss: {evaluations[-1][1]:.3f}, Critic Loss: {evaluations[-1][2]:.3f}"
    # np.save(f"./results/{file_name}", result_line)

    # 使用 torch.multiprocessing 并行训练任务
    manager = mp.Manager()
    replay_queue = manager.Queue()
    return_env_queue = manager.Queue()
    return_dict = manager.dict()
    
    # policy = policy.to('cpu')
    # policy_params = policy.state_dict()
    
    # Create env List
    # env_list = []
    # for i in range(args.num_processes):
    #     env = ManhattanTrafficEnv()
    #     env_list.append((i, env))

    # env_list = manager.list([(i, ManhattanTrafficEnv()) for i in range(args.num_processes)])
    # create pickle file for each env
    for i in range(args.num_processes):
        env = ManhattanTrafficEnv()
        process_seed = args.seed + i * 1000  # 每个环境使用不同的种子
        env.seed(process_seed)
        env.action_space.seed(process_seed)
        save_env_name = f"envs/env_{i}.pkl"
        with open(save_env_name, 'wb') as f:
            pickle.dump(env, f)
    
    
    # data_collection and training
    for cycle in range(args.cycles):
        # policy_params = torch.load('policy_checkpoint.pth')
        # multiprocessing
        processes = []
        seed_offsets = [seed_idx for seed_idx in range(args.num_processes)]
        if args.load_model == "":
            policy_flag = True if cycle > 0 else False
        else:
            policy_flag = True

        # TEMP: FOR BUFFER COLLECTION, POLICY FLAG SET AS FALSE
        # policy_flag = False

        for i in range(args.num_processes):
            p = mp.Process(target=collect_data, 
                         args=(i, args, seed_offsets[i], replay_queue, policy_flag, cycle))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # 从 replay_queue 中获取所有进程的数据并合并到一个 replay buffer 中
        while not replay_queue.empty():
            data = replay_queue.get()
            replay_buffer.merge(data)

        print(f"Replay buffer size: {replay_buffer.size}")
        if False: # shouldn't update the mean_std data during training
            if replay_buffer.size > 0:
                if cycle > 0:
                    mean, std = replay_buffer.get_mean_std()
                    if HALF_REQ:
                        np.savez('mean_std_data_NYC_HALF.npz', mean=mean, std=std)
                    else:
                        np.savez('mean_std_data_NYC_FULL.npz', mean=mean, std=std)
        
        # policy_params = train(args, policy_params, replay_buffer, cycle)

        # TEMP: FOR BUFFER COLLECTION, DONT TRAIN
        train(args, replay_buffer, cycle)

    # 最终评估并保存
    file_name = f"{args.policy}_{args.env}_{args.seed}_final"
    policy = policy.to(device)
    evaluations.append(eval_policy(policy, env, args.seed))
    # Create a list of result lines from all evaluations
    result_lines = [f"Reward: {eval[0]:.3f}, Actor Loss: {eval[1]:.3f}, Critic Loss: {eval[2]:.3f}, GNN Encoder Loss: {eval[3]:.3f}, Action: {eval[4]:.3f}" for eval in evaluations]
    # Save the result lines to a file
    np.save(f"./results/{file_name}", result_lines)
    if args.save_model:
        policy.save(f"./models/{file_name}")

def get_exploration_params(cycle, total_cycles):
    # 更平缓的策略选择概率增长
    policy_prob = max(0.4, min(0.8, 0.4 + 0.5 * (cycle / total_cycles)))
    # 噪声从0.3开始缓慢衰减到0.05
    noise = max(0.1, 0.4 * (1 - cycle / total_cycles))
    return policy_prob, noise

class FixedNormalizer(object):
    def __init__(self):
        # path = 'imac_200_mean_std_data.npz'
        if HALF_REQ:
            path = 'mean_std_data_50epcs.npz'
        else:
            path = 'mean_std_data_NYC_FULL.npz'

        # TEMP: FOR BUFFER COLLECTIO: SKIP NORMALIZATION
        assert os.path.exists(path), "mean_std not found"
        data = np.load(path)
        self.mean = torch.tensor(data['mean'], dtype=torch.float32).to(device)
        self.std = torch.tensor(data['std'], dtype=torch.float32).to(device)
        self.std[self.std == 0] = 1.0 # avoid division by zero

    def __call__(self, x):
        # TEMP: FOR BUFFER COLLECTIO: SKIP NORMALIZATION
        # return x
        return (x - self.mean) / (self.std)

class TrainingStats:
    def __init__(self):
        self.actor_losses = []
        self.critic_losses = []
        self.gnn_losses = []
        self.action_means = []
        self.action_stds = []
        self.reward_means = []
        self.saturated_actions = 0
        self.total_actions = 0
        
    def update(self, actor_loss, critic_loss, gnn_loss, action, reward):
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.gnn_losses.append(gnn_loss)
        self.action_means.append(np.mean(action))
        self.action_stds.append(np.std(action))
        self.reward_means.append(reward)
        
        # Track action saturation
        if abs(action) >= 0.48:  # 考虑接近边界的动作
            self.saturated_actions += 1
        self.total_actions += 1
        
    def get_recent_stats(self, window=100):
        return {
            'actor_loss': np.mean(self.actor_losses[-window:]),
            'critic_loss': np.mean(self.critic_losses[-window:]),
            'gnn_loss': np.mean(self.gnn_losses[-window:]),
            'action_mean': np.mean(self.action_means[-window:]),
            'action_std': np.mean(self.action_stds[-window:]),
            'reward_mean': np.mean(self.reward_means[-window:]),
            'saturation_rate': self.saturated_actions / max(1, self.total_actions)
        } 

if __name__ == '__main__':
    torch.set_num_threads(1)  # 限制每个进程的线程数
    main()
