import torch
from src.RL.models import GNN_Encoder, Actor, Critic, Normalizer, OUNoise
from src.RL.environment import ManhattanTrafficEnv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import os
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
    # print("Using CPU device")
# device = torch.device("cpu")
base_dir = 'saved_models'

if not os.path.exists(base_dir):
    os.makedirs(base_dir)
def train(models, environment, epochs):
    gnn_encoder, ddpg_agent = models
#   gnn_encoder = gnn_encoder.to(device)
#   ddpg_agent = ddpg_agent.to(device)
    system_time = 0.0
    while system_time < WARM_UP_DURATION:
        done = False
        warm_up_step = 0
        while not done:
            warm_up_step += 1
            environment.simulator.current_cycle_rej_rate = [] # reset the rejection rate to record the new cycle
            for step in tqdm(range(int(RL_STEP_LENGTH)), desc=f"WarmUP_{warm_up_step}"): # 2.5mins, 10 steps
                environment.simulator.system_time += TIME_STEP
                environment.simulator.run_cycle() # Run one cycle(15s)
                system_time = environment.simulator.system_time
            done = environment.warm_up_step()

    with open('training_log_1.txt', 'w') as log_file:
        for epoch in range(epochs):
            current_sim_time = environment.simulator.system_time
            last_req_ID = environment.simulator.last_req_ID
            environment.simulator.req_loader(current_sim_time, last_req_ID)

            # read reqs file from beginning
            # if environment.simulator.system_time > SIMULATION_DURATION:
            #     environment.simulator.system_time = 0
            #     environment.simulator.reset_veh_time()
            # state, network = environment.reset()

            network = environment.simulator.get_simulator_network_by_areas()
            current_state = None
            edge_index = graph_to_data(network)
            total_critic_loss = 0
            total_actor_loss = 0
            total_reward = 0
            done = False
            steps = 0

            while not done:
                environment.simulator.current_cycle_rej_rate = [] # reset the rejection rate to record the new cycle
                for step in tqdm(range(int(RL_STEP_LENGTH))): # 2.5mins, 10 steps
                    environment.simulator.system_time += TIME_STEP
                    environment.simulator.run_cycle() # Run one cycle(15s)
                
                if len(environment.past_rejections) < 3:
                    current_past_rej = 0.0
                    last_past_rej = 0.0
                else:
                    current_past_rej = environment.past_rejections[-2]
                    last_past_rej = environment.past_rejections[-3]

                if len(environment.past_thetas) < 3:
                    current_past_theta = 5.0
                    last_past_theta = 5.0
                else:
                    current_past_theta = environment.past_thetas[-2]
                    last_past_theta = environment.past_thetas[-3]
                #Record last state and current state
                if current_state == None:
                    last_state = torch.zeros((63,5))
                else:
                    last_state = current_state

                current_state = environment.simulator.get_simulator_state_by_areas()

                # cat_state = torch.cat((last_state, current_state), dim=1)

                current_x = current_state.clone().detach().requires_grad_(True)
                current_x = current_x.to(device)
                current_past_rej_tensor = torch.tensor([current_past_rej], dtype=torch.float32).to(device).unsqueeze(0).expand(current_x.size(0), -1)
                current_past_theta_tensor = torch.tensor([current_past_theta], dtype=torch.float32).to(device).unsqueeze(0).expand(current_x.size(0), -1)
                current_x = torch.cat((current_x, current_past_rej_tensor, current_past_theta_tensor), dim=1)
                current_graph_data = Data(x=current_x, edge_index=edge_index)

                last_x = last_state.clone().detach().requires_grad_(True)
                last_x = last_x.to(device)
                last_past_rej_tensor = torch.tensor([last_past_rej], dtype=torch.float32).to(device).unsqueeze(0).expand(last_x.size(0), -1)
                last_past_theta_tensor = torch.tensor([last_past_theta], dtype=torch.float32).to(device).unsqueeze(0).expand(last_x.size(0), -1)
                last_x = torch.cat((last_x, last_past_rej_tensor, last_past_theta_tensor), dim=1)
                last_graph_data = Data(x=last_x, edge_index=edge_index)

                current_state_encoded = gnn_encoder(current_graph_data)  # encode the state (1,32)
                last_state_encoded = gnn_encoder(last_graph_data)  # encode the state (1,32)

                current_action = ddpg_agent.select_action(current_state_encoded)
                reward, done, new_theta = environment.step(current_action)

                # Update the model, get the loss
                if len(environment.past_rewards) < 2:
                    last_reward = 0.0
                else:
                    last_reward = environment.past_rewards[-2]

                temp_env = environment.deepcopy()
                
                critic_loss, actor_loss = ddpg_agent.update_policy(last_reward, current_state_encoded, last_state_encoded)

                # Logging
                total_critic_loss += critic_loss
                total_actor_loss += actor_loss
                total_reward += reward
                ddpg_agent.total_steps += 1
                steps += 1

            avg_critic_loss = total_critic_loss / steps
            avg_actor_loss = total_actor_loss / steps
            avg_reward = total_reward / steps
            # Log the loss
            log_file.write(f"Epoch {epoch}: Avg Critic Loss = {avg_critic_loss}, Avg Actor Loss = {avg_actor_loss}, Avg Reward = {avg_reward}, Theta = {new_theta}\n")
            log_file.flush()
    
    # Save the models
    save_models(epoch, ddpg_agent)

def graph_to_data(network):
    # Convert the network to edge_index
    if isinstance(network, pd.DataFrame):
        network_array = network.values
    else:
        network_array = network

    # Find the indices of non-zero elements, which correspond to edges
    src_nodes, dst_nodes = np.nonzero(network_array)
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_index = edge_index.to(device)
    
    # Ensure the result is in the correct shape (2, num_edges)
    return edge_index

def save_models(epoch, model):
    actor_filename = os.path.join(base_dir, f'actor_{epoch}.pth')
    critic_filename = os.path.join(base_dir, f'critic_{epoch}.pth')
    actor_target_filename = os.path.join(base_dir, f'actor_target_{epoch}.pth')
    critic_target_filename = os.path.join(base_dir, f'critic_target_{epoch}.pth')

    torch.save(model.actor.state_dict(), actor_filename)
    torch.save(model.critic.state_dict(), critic_filename)
    torch.save(model.actor_target.state_dict(), actor_target_filename)
    torch.save(model.critic_target.state_dict(), critic_target_filename)
    print("Models saved successfully")

def load_models(epoch, model):
    model.actor.load_state_dict(torch.load(f'actor_{epoch}.pth'))
    model.critic.load_state_dict(torch.load(f'critic_{epoch}.pth'))
    model.actor_target.load_state_dict(torch.load(f'actor_target_{epoch}.pth'))
    model.critic_target.load_state_dict(torch.load(f'critic_target_{epoch}.pth'))
    print("Models loaded successfully")

def warm_up_train(models, environment, epochs = WARM_UP_EPOCHS):
    gnn_encoder, ddpg_agent = models

    for epoch in range(epochs):
        state, network = environment.reset()
        done = False
        warm_up_step = 0
        while not done:
            warm_up_step += 1
            for step in tqdm(range(int(RL_STEP_LENGTH)), desc=f"WarmUP_{warm_up_step}"): # 2.5mins, 10 steps
                environment.simulator.system_time += TIME_STEP
                environment.simulator.run_cycle() # Run one cycle(15s)
            done, past_rejections = environment.warm_up_step()
    return past_rejections

def req_loader(current_sim_time, last_req_ID):
    PATH_REQUESTS = f"{ROOT_PATH}/NYC/NYC_Andres_data/"
    FILE_NAME = "NYC_Manhattan_Requests_size3_day"

    random_day = np.random.randint(1, 11)
    SELECTED_FILE = FILE_NAME + str(random_day) + '.csv'
    TEMP_FILE_NAME = 'temp_req.csv'

    with open(os.path.join(PATH_REQUESTS, SELECTED_FILE), 'r') as f:
        temp_req_matrix = pd.read_csv(f)
    temp_req_matrix['ReqID'] += last_req_ID
    temp_req_matrix['ReqTime'] += current_sim_time
    temp_req_matrix.to_csv(os.path.join(PATH_REQUESTS, TEMP_FILE_NAME))


if __name__ == "__main__":
    # Initialize environment
    environment = ManhattanTrafficEnv()
    
    # Setup dimensions
    state_dim = 32  # Assume output dimension from GNN
    action_dim = environment.action_space.shape[0]
    max_action = float(environment.action_space.high[0])

    # Initialize models
    gnn_encoder = GNN_Encoder(num_features=NUM_FEATURES, hidden_dim=64, output_dim=state_dim)
    actor = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    critic = Critic(state_dim=state_dim, action_dim=action_dim)

    # Setup optimizer
    optimizer = torch.optim.Adam(
        list(gnn_encoder.parameters()) + list(actor.parameters()) + list(critic.parameters()), lr=1e-4
    )

    # Train the model
    train(gnn_encoder, actor, critic, environment, epochs=1000, optimizer=optimizer)
