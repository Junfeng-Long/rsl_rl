#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


class DynamicsModel(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_actions,
        dynamics_model_hidden_dims=[512, 512, 512],
        activation="elu",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        input_dim = num_actor_obs + num_actions
        output_dim = num_actor_obs
        # Dynamics Model
        dynamics_model_layers = []
        dynamics_model_layers.append(nn.Linear(input_dim, dynamics_model_hidden_dims[0]))
        dynamics_model_layers.append(activation)
        for layer_index in range(len(dynamics_model_hidden_dims)):
            if layer_index == len(dynamics_model_hidden_dims) - 1:
                dynamics_model_layers.append(nn.Linear(dynamics_model_hidden_dims[layer_index], output_dim))
            else:
                dynamics_model_layers.append(nn.Linear(dynamics_model_hidden_dims[layer_index], dynamics_model_hidden_dims[layer_index + 1]))
                dynamics_model_layers.append(activation)
        self.dynamics_model = nn.Sequential(*dynamics_model_layers)
        # Optimizer
        self.optimizer = optim.Adam(self.dynamics_model.parameters(), lr=1e-5)
        self.loss = 0

        print(f"Dynamics Model: {self.dynamics_model}")

    def forward(self, observations, actions):
        return self.dynamics_model(torch.cat([observations, actions], dim=-1))
    
    # def accumulate_loss(self, observations, actions, next_observations):
    #     predicted_next_observations = self.forward(observations, actions)
    #     self.loss += nn.MSELoss(reduction='sum')(predicted_next_observations, next_observations).item()
        
    # def optimize(self):
    #     self.optimizer.zero_grad()
    #     self.loss.backward()
    #     self.optimizer.step()
    #     loss = self.loss.item()
    #     self.loss = 0
    #     return loss
    
    def optimize(self, observations, actions, next_observations):
        self.optimizer.zero_grad()
        predicted_next_observations = self.forward(observations, actions)
        loss = nn.MSELoss()(predicted_next_observations, next_observations)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
