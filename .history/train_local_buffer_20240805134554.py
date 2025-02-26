import numpy as np
import torch
import gym
import argparse
import os

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

num_gpus = torch.cuda.device_count()

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

def train(args, replay_buffer):
    print("---------------------------------------")
    print(f"Training Start")
    print("---------------------------------------")
    if device == torch.device("cuda"):
        torch.cuda.set_device(0)

    # 初始化策略
    env = ManhattanTrafficEnv()
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
    policy = policy.to(device)

    network = env.simulator.get_simulator_network_by_areas()
    edge_index = env.graph_to_data(network)

    batch_size = 128
    total_epochs = int(1e6)

    for epoch in range(total_epochs):
        policy.train(replay_buffer, batch_size, edge_index)
        if epoch % args.eval_freq == 0:
            avg_reward, avg_actor_loss, avg_critic_loss, avg_gnn_encoder_loss, avg_action = eval_policy(policy, env, args.seed)
            with open('training_log.txt', 'a') as log_file:
                log_file.write(f"Epoch:{epoch} | Actor_Loss:{avg_actor_loss:.3f} | Critic_Loss:{avg_critic_loss:.3f} | GNN_Encoder_Loss:{avg_gnn_encoder_loss:.3f} | Avg_Reward:{avg_reward:.3f} | Avg_Action:{avg_action:.3f}\n")
        if epoch % 100 == 0:
            policy.save(f"./saved_models/TD3_EPOCH_{epoch}")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="ManhattanTrafficEnv")     # Doesn't matter. Overriden by ManhattanTrafficEnv
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    # parser.add_argument("--start_timesteps", default=100, type=int) # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=50, type=int)       # How often (time steps) we evaluate
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
    parser.add_argument("--num_processes", default=4, type=int)     # Number of processes to use 
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

    # Initialize policy
    if args.policy == "TD3":
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

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
    replay_buffer.load('replay_buffer_data_100800.npz')

    # Evaluate untrained policy
    evaluations = []

    print(f"Replay buffer size: {replay_buffer.size}")
  
    # policy_params = train(args, policy_params, replay_buffer, cycle)
    train(args, replay_buffer)

    # 最终评估并保存
    file_name = f"{args.policy}_{args.env}_{args.seed}_final"
    policy = policy.to(device)
    evaluations.append(eval_policy(policy, 
                                   , args.seed))
    # Create a list of result lines from all evaluations
    result_lines = [f"Reward: {eval[0]:.3f}, Actor Loss: {eval[1]:.3f}, Critic Loss: {eval[2]:.3f}, GNN Encoder Loss: {eval[3]:.3f}, Action: {eval[4]:.3f}" for eval in evaluations]
    # Save the result lines to a file
    np.save(f"./results/{file_name}", result_lines)
    if args.save_model:
        policy.save(f"./models/{file_name}")


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

if __name__ == '__main__':
    main()

