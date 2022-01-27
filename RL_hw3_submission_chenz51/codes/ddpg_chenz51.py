"""
Implementation of DDPG agent

Author: Zirong Chen
"""

from agent import Agent
from replay_buffer import OffPolicyReplayBuffer
import torch
import torch.nn as nn
from utils import *
from torch.optim import Adam
from ddpg_networks import *
from copy import deepcopy


class DDPGAgent(Agent):

    def __init__(self, params):
        super().__init__()

        self.params = params

        self.device = self.params["device"]
        self.obs_dim = self.params["obs_dim"]
        self.act_dim = self.params["act_dim"]
        self.act_low = self.params["act_low"]
        self.act_high = self.params["act_high"]
        self.batch_size = self.params["batch_size"]
        self.replay_buffer = self.params["replay_buffer"]
        self.gamma = self.params["gamma"]
        self.noise_scale = self.params["noise_scale"]
        self.polyak = self.params["polyak"]
        self.pi_lr = self.params["pi_lr"]
        self.q_lr = self.params["q_lr"]

        self.actor_critic = MLPActorCritic(act_dim=self.act_dim,
                                           obs_dim=self.obs_dim,
                                           act_limit=self.act_high).to(self.device)

        # self.actor_critic_target = deepcopy(self.actor_critic)

        self.actor_critic_target = MLPActorCritic(act_dim=self.act_dim,
                                                  obs_dim=self.obs_dim,
                                                  act_limit=self.act_high).to(self.device)

        self.pi_optim = Adam(self.actor_critic.pi.parameters(), lr=self.pi_lr)
        self.q_optim = Adam(self.actor_critic.q.parameters(), lr=self.q_lr)

        # Freeze the target net, update later using ployak averaging
        for param in self.actor_critic_target.parameters():
            param.requires_grad = False

    def train(self, *args):
        batch = self.replay_buffer.sample_batch(self.batch_size)

        # First run one gradient descent step for Q
        self.q_optim.zero_grad()
        loss_q = self.compute_loss_q(batch)
        loss_q.backward()
        self.q_optim.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step
        for p in self.actor_critic.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optim.zero_grad()
        loss_pi = self.compute_loss_pi(batch)
        loss_pi.backward()
        self.pi_optim.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step
        for p in self.actor_critic.parameters():
            p.requires_grad = True

        with torch.no_grad():
            for param, param_targ in zip(self.actor_critic.parameters(), self.actor_critic_target.parameters()):
                param_targ.data.mul_(self.polyak)
                param_targ.data.add_((1 - self.polyak) * param.data)

        return loss_q, loss_pi

    def choose_action(self, obs):
        obs = obs.to(self.device)
        action = self.actor_critic.act(obs)
        action += self.noise_scale * np.random.randn(self.act_dim)
        return np.clip(action, self.act_low, self.act_high)

    def compute_loss_pi(self, data):
        obs = data["obs"]
        q_pi = self.actor_critic.q(obs, self.actor_critic.pi(obs))
        loss_pi = -q_pi.mean()

        return loss_pi

    def compute_loss_q(self, data):
        obs, act, rew, next_obs, done = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q_val = self.actor_critic.q(obs, act)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.actor_critic_target.q(next_obs, self.actor_critic_target.pi(next_obs))
            backup = rew + self.gamma * (1 - done) * q_pi_targ

        loss_q = ((q_val - backup) ** 2).mean()

        return loss_q