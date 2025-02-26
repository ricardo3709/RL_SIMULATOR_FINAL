import numpy as np
import torch
from mpi4py import MPI
import gym

# MPI 初始化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 假设 max_action 和 action_dim 是预定义的变量
max_action = 1.0  # 示例值，请根据你的实际设置进行调整
action_dim = 3  # 示例值，请根据你的实际设置进行调整

# 假设 args 包含 start_timesteps, max_timesteps, eval_freq, save_model 等属性
class Args:
    start_timesteps = 1
    max_timesteps = 1000000
    eval_freq = 5000
    save_model = True
    env = "Pendulum-v0"  # 示例环境
    seed = 0
    expl_noise = 0.1
    batch_size = 100

args = Args()

# 初始化环境和策略
env = gym.make(args.env)
env.seed(args.seed + rank)
torch.manual_seed(args.seed + rank)
np.random.seed(args.seed + rank)

# 创建ReplayBuffer
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_size = 1000000
replay_buffer = ReplayBuffer(max_size, state_dim, action_dim)

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.next_state[self.ptr] = next_state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.done[ind]).to(device)
        )

    def get_all_data(self):
        return {
            'state': self.state[:self.size],
            'next_state': self.next_state[:self.size],
            'action': self.action[:self.size],
            'reward': self.reward[:self.size],
            'done': self.done[:self.size]
        }

    def merge(self, data):
        data_size = len(data['state'])
        insert_size = min(data_size, self.max_size - self.size)
        self.state[self.ptr:self.ptr + insert_size] = data['state'][:insert_size]
        self.next_state[self.ptr:self.ptr + insert_size] = data['next_state'][:insert_size]
        self.action[self.ptr:self.ptr + insert_size] = data['action'][:insert_size]
        self.reward[self.ptr:self.ptr + insert_size] = data['reward'][:insert_size]
        self.done[self.ptr:self.ptr + insert_size] = data['done'][:insert_size]
        self.ptr = (self.ptr + insert_size) % self.max_size
        self.size = min(self.size + insert_size, self.max_size)

# 假设 policy 是一个包含 select_action 方法的对象
class Policy:
    def __init__(self):
        self.actor = torch.nn.Linear(state_dim, action_dim).to(device)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size):
        pass  # 这里填入你的训练逻辑

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename)

policy = Policy()

# 训练函数
def train(env, policy, replay_buffer, max_action, action_dim, args):
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)  # need to check done definition
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps and rank == 0:
            policy.train(replay_buffer, args.batch_size)

        if done:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0 and rank == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model:
                policy.save(f"./models/{file_name}")

    # 每个进程在最后发送本地 replay buffer 数据到主进程
    local_data = replay_buffer.get_all_data()
    gathered_data = comm.gather(local_data, root=0)

    if rank == 0:
        for data in gathered_data:
            replay_buffer.merge(data)

# 每个进程独立运行 train
train(env, policy, replay_buffer, max_action, action_dim, args)

# MPI Finalize
comm.Barrier()
MPI.Finalize()