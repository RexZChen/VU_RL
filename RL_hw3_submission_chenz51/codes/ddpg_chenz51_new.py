"""
Implementation of DDPG agent
Main difference: different buffer, different forward

Author: Zirong Chen
"""
from agent import Agent
from utils import mlp
from torch.optim import Adam
from torch.nn import MSELoss
import numpy as np
import torch
from ddpg_networks_new import *


class DDPGAgent_new(Agent):

    """
    DDPG agent for continuous action space
    """

    def __init__(self, agent_params):
        super(DDPGAgent_new, self).__init__()
        self.agent_params = agent_params
        self.obs_dim = agent_params["obs_dim"]
        self.act_dim = agent_params["act_dim"]
        self.act_high = agent_params["act_high"]
        self.act_low = agent_params["act_low"]
        self.hidden_sizes = agent_params["hidden_sizes"]
        self.batch_size = agent_params["batch_size"]
        self.replay_buffer = agent_params["replay_buffer"]
        self.device = agent_params["device"]
        self.gamma = agent_params["gamma"]
        self.polyak = agent_params["polyak"]
        self.noise_scale = agent_params["noise_scale"]
        self.actor = MLPActorCritic(self.obs_dim, self.act_dim, self.act_high, self.hidden_sizes).to(self.device)
        self.actor_target = MLPActorCritic(self.obs_dim, self.act_dim, self.act_high, self.hidden_sizes).to(self.device)
        self.pi_optimizer = Adam(self.actor.pi_network.parameters(), lr=agent_params["pi_lr"])
        self.q_optimizer = Adam(self.actor.q_network.parameters(), lr=agent_params["q_lr"])
        self.mse_loss_func = MSELoss()
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for param in self.actor_target.parameters():
            param.requires_grad = False

    def choose_action(self, obs, *args):
        obs = obs.to(self.device)
        action = self.actor.act(obs).cpu().numpy()
        action += self.noise_scale * np.random.randn(self.act_dim)
        return np.clip(action, self.act_low, self.act_high)

    def train(self, *args):
        states, actions, rewards, dones, next_states = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.device)  # (batch, obs_dim)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)  # (batch)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.int).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)

        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q((states, actions, rewards, dones, next_states))
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for param in self.actor.q_network.parameters():
            param.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(states)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for param in self.actor.q_network.parameters():
            param.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for param, param_targ in zip(self.actor.parameters(), self.actor_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                param_targ.data.mul_(self.polyak)
                param_targ.data.add_((1 - self.polyak) * param.data)

        return loss_q.item(), loss_pi.item()

    def compute_loss_q(self, data):
        states, actions, rewards, dones, next_states = data
        q_value = self.actor.q_network(states, actions)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.actor_target.q_network(next_states, self.actor_target.pi_network(next_states))
            backup = rewards + self.gamma * (1 - dones) * q_pi_targ

        # MSE loss against Bellman backup
        # loss_q = ((q_value - backup) ** 2).mean()
        loss_q = self.mse_loss_func(q_value, backup)

        return loss_q

    def compute_loss_pi(self, obs):  # obs needs to be torch tensor
        act = self.actor.pi_network(obs)
        q_pi = self.actor.q_network(obs, act)
        return -q_pi.mean()