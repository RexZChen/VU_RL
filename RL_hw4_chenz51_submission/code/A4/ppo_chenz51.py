from agent import Agent
from ppo_networks import MLPActorCritic
import torch
from torch.optim import Adam
from torch.nn import MSELoss


class PPOAgent(Agent):
    def __init__(self, agent_params):
        super(PPOAgent, self).__init__()

        self.agent_params = agent_params
        self.obs_dim = self.agent_params["obs_dim"]
        self.act_dim = self.agent_params["act_dim"]
        self.gamma = self.agent_params["gamma"]
        self.lam = self.agent_params["lambda"]
        self.clip_ratio = self.agent_params["clip_ratio"]
        self.entropy_coef = self.agent_params["entropy_coef"]
        self.train_v_iters = self.agent_params["train_v_iters"]
        self.train_pi_iters = self.agent_params["train_pi_iters"]
        self.replay_buffer = self.agent_params["replay_buffer"]
        self.mse_loss_func = MSELoss()

        self.actor_critic = MLPActorCritic(self.obs_dim, self.act_dim, agent_params["hidden_sizes"])

        self.pi_optim = Adam(self.actor_critic.pi_network.parameters(), lr=self.agent_params["pi_lr"])
        self.v_optim = Adam(self.actor_critic.v_network.parameters(), lr=self.agent_params["vf_lr"])

    def choose_action(self, obs, *args):
        with torch.no_grad():
            act, log_prob = self.actor_critic.pi_network(obs)  # obs: [1, obs_dim]
            # act needs to be either a vector(continuous action space) or a scalar(discrete action space)
            return torch.squeeze(act, dim=0).cpu().numpy(), log_prob.item(), torch.squeeze(self.actor_critic.v_network(obs), dim=1).item()

    def compute_loss_pi(self, data):
        logp_a_new = self.actor_critic.pi_network.log_prob_from_distribution(data["obs"], data["act"])
        entropy = self.actor_critic.pi_network.entropy_from_distribution(data["obs"])
        logp_a_old = data["log_prob"]
        ratio = torch.exp(logp_a_new - logp_a_old)
        clamped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        ratio_adv = ratio * data["gae"]
        clamped_adv = clamped_ratio * data["gae"]
        loss = -torch.mean(torch.min(ratio_adv, clamped_adv)) - self.entropy_coef * entropy
        return loss

    def compute_loss_v(self, data):
        predicted_values = torch.squeeze(self.actor_critic.v_network(data['obs']), dim=1)
        return self.mse_loss_func(predicted_values, data['return'])

    def train(self, *args):

        assert self.train_v_iters > 0 and self.train_pi_iters > 0

        data_cpu = self.replay_buffer.get_data()
        data = {k: data_cpu[k].to("cpu") for k in data_cpu}

        pi_l_old = self.compute_loss_pi(data).item()
        v_l_old = self.compute_loss_v(data).item()

        for _ in range(self.train_pi_iters):
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
