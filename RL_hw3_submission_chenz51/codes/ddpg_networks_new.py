"""
Networks for DDPG: (1) MLPActor; (2) MLPCritic; (3) MLPActorCritic
Main difference: different buffer, different forward

Author: Zirong Chen
"""
import torch
import torch.nn as nn
from utils import *


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super(MLPActor, self).__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp_ddpg(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)


class MLPQNetwork(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(MLPQNetwork, self).__init__()
        self.q_network = mlp_ddpg([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q_value = self.q_network(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q_value, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, act_high, hidden_sizes, activation=nn.Tanh):
        super(MLPActorCritic, self).__init__()

        # build policy and value functions
        self.pi_network = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_high)
        self.q_network = MLPQNetwork(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi_network(obs)