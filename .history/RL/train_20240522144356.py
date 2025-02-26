import torch
from models import GNN_Encoder, Actor, Critic
from environment import ManhattanTrafficEnv
from torch_geometric.data import Data
import numpy as np

def train(model, environment, epochs):
    for epoch in range(epochs):
        state = environment.reset()
        total_critic_loss = 0
        total_actor_loss = 0
        done = False
        steps = 0

        while not done:
            action = model.select_action(state)
            next_state, reward, done, _ = environment.step(action)

            # Update the model, get the loss
            critic_loss, actor_loss = model.update_policy(state, action, reward, next_state)

            # Logging
            total_critic_loss += critic_loss
            total_actor_loss += actor_loss
            model.total_steps += 1
            steps += 1
            state = next_state

        avg_critic_loss = total_critic_loss / steps
        avg_actor_loss = total_actor_loss / steps
        print(f"Epoch {epoch}: Avg Critic Loss = {avg_critic_loss}, Avg Actor Loss = {avg_actor_loss}")


if __name__ == "__main__":
    # Initialize environment
    environment = ManhattanTrafficEnv()
    
    # Setup dimensions
    state_dim = 32  # Assume output dimension from GNN
    action_dim = environment.action_space.shape[0]
    max_action = float(environment.action_space.high[0])

    # Initialize models
    gnn_encoder = GNN_Encoder(num_features=4, hidden_dim=64, output_dim=state_dim)
    actor = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    critic = Critic(state_dim=state_dim, action_dim=action_dim)

    # Setup optimizer
    optimizer = torch.optim.Adam(
        list(gnn_encoder.parameters()) + list(actor.parameters()) + list(critic.parameters()), lr=1e-4
    )

    # Train the model
    train(gnn_encoder, actor, critic, environment, epochs=100, optimizer=optimizer)
