import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import mlp
from torch.distributions.normal import Normal


class MLPGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh):
        super(MLPGaussianActor, self).__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=True)
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        dist = self._distribution(obs)
        act = dist.sample()
        log_prob = torch.sum(dist.log_prob(act), dim=-1)
        return act, log_prob

    def log_prob_from_distribution(self, obs, act):
        dist = self._distribution(obs)
        return torch.sum(dist.log_prob(act), dim=-1)

    def entropy_from_distribution(self, obs):
        dist = self._distribution(obs)
        return dist.entropy().mean()

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)


class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh):
        super(MLPActorCritic, self).__init__()

        # build policy and value functions
        self.pi_network = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        sizes = [obs_dim] + hidden_sizes + [1]
        self.v_network = mlp(sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi_network(obs)