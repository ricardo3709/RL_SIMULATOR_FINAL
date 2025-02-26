import torch
from models import GNN_Encoder, Actor, Critic
from environment import ManhattanTrafficEnv
from torch_geometric.data import Data

def data_preparation(state):
    [idle_cars, rebalancing_cars, request_gen_counts, request_rej_counts] = state
    idle_cars = torch.tensor(idle_cars)  # [4091]
    rebalancing_cars = torch.tensor(rebalancing_cars)  # [4091]
    request_gen_counts = torch.tensor(request_gen_counts)  # [4091]
    request_rej_counts = torch.tensor(request_rej_counts)  # [4091]

    # combine all features
    features = torch.stack((idle_cars, rebalancing_cars, request_gen_counts, request_rej_counts), dim=1)


def train(model, environment, epochs, optimizer):
    for epoch in range(epochs):
        state = environment.reset()
        total_loss = 0
        while not done:
            action = model.select_action(state)
            next_state, reward, done, _ = environment.step(action)
            loss = model.update(state, action, reward, next_state)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state
            total_loss += loss.item()
        print(f"Epoch {epoch}: Total Loss: {total_loss}")

if __name__ == "__main__":
    # Initialize environment and model
    environment = ManhattanTrafficEnv()
    gnn_encoder = GNN_Encoder(num_features=..., hidden_dim=..., output_dim=...)
    actor = Actor(state_dim=..., action_dim=..., max_action=...)
    critic = Critic(state_dim=..., action_dim=...)

    # Setup optimizer (example)
    optimizer = torch.optim.Adam(list(gnn_encoder.parameters()) + list(actor.parameters()) + list(critic.parameters()), lr=1e-4)

    # Train the model
    train(gnn_encoder, actor, critic, environment, epochs=100, optimizer=optimizer)
