"""
Implementation of REINFORCE agent

Author: Zirong Chen
"""

from agent import Agent
from reinforce_networks import *
from replay_buffer import OnPolicyBuffer
from utils import *
import torch
import torch.nn as nn
from torch.optim import Adam


class ReinforceAgent(Agent):

    def __init__(self, params):
        super(ReinforceAgent, self).__init__()

        self.params = params

        self.device = self.params["device"]
        self.obs_dim = self.params["obs_dim"]
        self.act_dim = self.params["act_dim"]
        self.gamma = self.params["gamma"]
        self.lam = self.params["lam"]
        self.train_v_iters = self.params["train_v_iters"]
        self.pi_lr = self.params["pi_lr"]
        self.vf_lr = self.params["v_lr"]

        self.replay_buffer = self.params["replay_buffer"]

        self.actor_critic = MLPActorCritic(self.obs_dim, self.act_dim).to(self.device)

        self.pi_optim = Adam(self.actor_critic.pi.parameters(), lr=self.pi_lr)
        self.v_optim = Adam(self.actor_critic.v.parameters(), lr=self.vf_lr)

    def compute_loss_pi(self, data):
        obs, act, adv = data['obs'], data['act'], data['adv']

        pi, logp = self.actor_critic.pi(obs, act)
        loss_pi = -(logp * adv).mean()

        return loss_pi

    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.actor_critic.v(obs) - ret) ** 2).mean()

    def train(self, *args):
        data = self.replay_buffer.get()

        pi_l_old = self.compute_loss_pi(data).item()
        v_l_old = self.compute_loss_v(data).item()

        self.pi_optim.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optim.step()

        for _ in range(self.train_v_iters):
            self.v_optim.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.v_optim.step()

        return pi_l_old, v_l_old