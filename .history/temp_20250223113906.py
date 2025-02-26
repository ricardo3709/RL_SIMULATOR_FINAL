def collect_data(rank, args, seed_offset, process_queue, policy_flag, cycle):
    # ... 前面的代码保持不变，直到 replay_queue 的使用部分 ...

    # 不再使用共享的 replay_queue，而是使用 process_queue
    try:
        save_env_name = f"envs/env_{rank}.pkl"
        with open(save_env_name, 'rb') as f:
            env = pickle.load(f)
    except Exception as e:
        print(f"Process {rank} - Error loading environment: {e}")
        return

    # ... 中间的环境初始化和数据收集逻辑不变 ...

    # 将数据写入进程自己的队列
    process_queue.put(replay_buffer.get_all_data())

    try:
        with open(save_env_name, 'wb') as f:
            pickle.dump(env, f)
    except Exception as e:
        print(f"Process {rank} - Error saving environment: {e}")

def main():
    # ... 前面的初始化代码不变 ...

    replay_buffer_max_size = 50 * args.num_processes * args.data_collection_step
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(replay_buffer_max_size))

    # 使用 Manager 创建进程管理器
    manager = mp.Manager()
    
    # 为每个进程创建独立的 Queue
    process_queues = [manager.Queue(maxsize=100) for _ in range(args.num_processes)]

    # 创建环境 pickle 文件
    for i in range(args.num_processes):
        env = ManhattanTrafficEnv()
        process_seed = args.seed + i * 1000
        env.seed(process_seed)
        env.action_space.seed(process_seed)
        save_env_name = f"envs/env_{i}.pkl"
        with open(save_env_name, 'wb') as f:
            pickle.dump(env, f)

    # 数据收集和训练循环
    for cycle in range(args.cycles):
        processes = []
        seed_offsets = [seed_idx for seed_idx in range(args.num_processes)]
        policy_flag = True if (args.load_model != "" or cycle > 1) else False

        # 启动进程，每个进程使用自己的队列
        for i in range(args.num_processes):
            p = mp.Process(target=collect_data, 
                           args=(i, args, seed_offsets[i], process_queues[i], policy_flag, cycle))
            p.start()
            processes.append(p)

        # 在 join 前从每个队列中收集数据
        for i, q in enumerate(process_queues):
            try:
                while not q.empty():
                    data = q.get(timeout=10)  # 设置超时，避免无限等待
                    replay_buffer.merge(data)
            except Exception as e:
                print(f"Error getting data from queue {i}: {e}")

        # 等待所有进程完成
        for p in processes:
            p.join(timeout=3600)
            if p.is_alive():
                print(f"Process {p.pid} timed out, terminating...")
                p.terminate()

        # 再次确保所有队列的数据被清空
        for i, q in enumerate(process_queues):
            try:
                while not q.empty():
                    data = q.get(timeout=10)
                    replay_buffer.merge(data)
            except Exception as e:
                print(f"Error clearing queue {i}: {e}")

        print(f"Replay buffer size: {replay_buffer.size}")
        train(args, replay_buffer, cycle)

    # ... 后面的评估和保存代码不变 ...