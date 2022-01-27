import numpy as np
import torch


class OnPolicyBuffer:

    def __init__(self, gamma, lam, obs_dim, act_dim, size):
        self.gamma = gamma  # discount factor
        self.lam = lam  # lambda for computing GAE value
        self.obs_buffer = np.zeros((size, obs_dim))
        self.act_buffer = np.zeros((size, act_dim))
        self.reward_buffer = np.zeros(size)
        self.v_buffer = np.zeros(size)  # V value
        self.return_buffer = np.zeros(size)  # G(discounted return)
        self.adv_buffer = np.zeros(size)  # Advantage value
        self.gae_buffer = np.zeros(size)  # GAE value
        self.log_prob_buffer = np.zeros(size)
        self.current_start_index = 0
        self.index = 0

    def add_transition(self, obs, act, reward, value, log_prob):
        self.obs_buffer[self.index] = obs
        self.act_buffer[self.index] = act
        self.reward_buffer[self.index] = reward
        self.v_buffer[self.index] = value
        self.log_prob_buffer[self.index] = log_prob
        self.index += 1

    def finish_episode(self, last_v):
        """
        traverse backwards to fill G, A and GAE values
        """
        g, gae = last_v, 0
        for i in range(self.index - 1, self.current_start_index - 1, -1):
            # discounted return
            g = self.reward_buffer[i] + self.gamma * g
            # adv value
            self.return_buffer[i], self.adv_buffer[i] = g, g - self.v_buffer[i]
            # gae value
            delta = self.reward_buffer[i] + self.gamma * last_v - self.v_buffer[i]
            gae = delta + self.gamma * self.lam * gae
            self.gae_buffer[i], last_v = gae, self.v_buffer[i]
        self.current_start_index = self.index

    def get_data(self):

        self._evict_data()

        data_map = {
            "obs": torch.tensor(self.obs_buffer, dtype=torch.float32),
            "act": torch.tensor(self.act_buffer),
            "return": torch.tensor(self.return_buffer, dtype=torch.float32),
            "adv": torch.tensor(self.adv_buffer, dtype=torch.float32),
            "v": torch.tensor(self.v_buffer, dtype=torch.float32),
            "gae": torch.tensor(self.gae_buffer, dtype=torch.float32),
            "log_prob": torch.tensor(self.log_prob_buffer, dtype=torch.float32)
        }
        return data_map

    def _evict_data(self):
        """reset index"""
        self.index, self.current_start_index = 0, 0