import numpy as np
import torch
import gym
import argparse
import os
from mpi4py import MPI

import src.RL.utils as utils
import src.RL.TD3 as TD3
from src.RL.environment import ManhattanTrafficEnv
from src.simulator.config import *
from tqdm import tqdm

# Check device availability: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    # print("Using CUDA device")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    # print("Using MPS device")
else:
    device = torch.device("cpu")

# MPI 初始化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, seed, eval_episodes=1):
    eval_env.seed(seed + 100)
    avg_reward = 0.
    actor_losses = []
    critic_losses = []
    # Warm up the agent before start evaluation
    system_time = 0.0
    while system_time < WARM_UP_DURATION:
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

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        state = state.to(device)
        while not done:
            action = policy.select_action(np.array(state.detach().cpu()))
            state, reward, done, _ = eval_env.step(action)
            state = state.to(device)
            actor_loss, critic_loss = policy.get_loss()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            avg_reward += reward
            done = True

    avg_reward /= eval_episodes
    avg_actor_loss = np.mean(actor_losses)
    avg_critic_loss = np.mean(critic_losses)

    if rank == 0:  # Only the main process prints the evaluation result
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print(f"Average Actor Loss: {avg_actor_loss:.3f}")
        print(f"Average Critic Loss: {avg_critic_loss:.3f}")
        print("---------------------------------------")
    return avg_reward, avg_actor_loss, avg_critic_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="ManhattanTrafficEnv")     # Doesn't matter. Overriden by ManhattanTrafficEnv
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    if rank == 0:
        print("---------------------------------------")
        print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")

        if not os.path.exists("./results"):
            os.makedirs("./results")

        if args.save_model and not os.path.exists("./models"):
            os.makedirs("./models")

    # 初始化环境和策略
    env = ManhattanTrafficEnv()
    env.seed(args.seed + rank)
    env.action_space.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

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
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    if rank == 0:
        evaluations = [eval_policy(policy, env, args.seed)]
        np.save(f"./results/{file_name}", evaluations)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Warm up the agent before start training
    system_time = 0.0
    while system_time < WARM_UP_DURATION:
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

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state.cpu().detach().numpy()))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)  # need to check done definition
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Merge replay buffer data from all processes
        if t >= args.start_timesteps:
            # 每个进程在最后发送本地 replay buffer 数据到主进程
            local_data = replay_buffer.get_all_data()
            gathered_data = comm.gather(local_data, root=0)

            if rank == 0:
                for data in gathered_data:
                    replay_buffer.merge(data)
                # Train agent after collecting sufficient data
                policy.train(replay_buffer, args.batch_size)

        if done:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            if rank == 0:
                evaluations.append(eval_policy(policy, env, args.seed))
                result_lines = [f"Reward: {eval[0]:.3f}, Actor Loss: {eval[1]:.3f}, Critic Loss: {eval[2]:.3f}" for eval in evaluations]
                np.save(f"./results/{file_name}", evaluations)
                if args.save_model:
                    policy.save(f"./models/{file_name}")

    # MPI Finalize
    comm.Barrier()
    MPI.Finalize()