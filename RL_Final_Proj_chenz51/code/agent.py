from torch.optim import Adam
import scipy.signal
import torch
import numpy as np
from replay_buffer import OnPolicyBuffer
from policy_agent import PolicyAgent


class MetaLearnerAgent:

    def __init__(self,
                 policy_agent: PolicyAgent,
                 gamma,
                 lam,
                 clip_ratio,
                 c1,
                 meta_lr,
                 standardize_advantages=True):
        self.policy_agent = policy_agent
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.c1 = c1
        self.standardize_advantages = standardize_advantages
        self.meta_lr = meta_lr
        self.meta_optimizer = Adam(self.policy_agent.parameters(), self.meta_lr)

    def adapt(self, alpha, buffer, params):
        obs, acts, concatenated_rewards, unconcatenated_rewards, _, _ = buffer.get_data()
        q_values = self.discount_cumsum(concatenated_rewards, self.gamma)
        advantage_values = self.calculate_advantages(obs, q_values, params)

        pi, logp_a = self.policy_agent.get_pi(
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(acts, dtype=torch.float32),
            params
        )

        loss_pi = -(logp_a * torch.as_tensor(advantage_values, dtype=torch.float32)).mean()
        targets_n = torch.as_tensor((q_values - np.mean(q_values)) / (np.std(q_values) + 1e-8), dtype=torch.float32)
        predicted_n = self.policy_agent.get_value(torch.as_tensor(obs, dtype=torch.float32), params)

        loss_v = ((predicted_n - targets_n) ** 2).mean()
        grads = torch.autograd.grad(loss_pi + 0.5 * loss_v, [i[1] for i in params.items()])

        updated_params = self.policy_agent.update_params(grads, alpha)
        return updated_params

    def step(self, adapted_params_list, meta_paths_list):
        losses = []
        for adapted_params, meta_paths in zip(adapted_params_list, meta_paths_list):
            loss = self.outer_loss(meta_paths, adapted_params)
            losses.append(loss)

        loss = torch.stack(losses).mean()

        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()

        return loss.item()

    def outer_loss(self, meta_paths, adapted_params):
        buffer = OnPolicyBuffer()
        buffer.add_rollouts(meta_paths)
        obs, acts, concatenated_rewards, unconcatenated_rewards, concatenated_logp_as, _ = buffer.get_data()
        q_values = self.discount_cumsum(concatenated_rewards, self.gamma)
        gae_values = self.calculate_gae(concatenated_rewards, obs, adapted_params, self.lam)
        pi, logp_a = self.policy_agent.get_pi(
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(acts, dtype=torch.float32),
            adapted_params
        )
        ratio = torch.exp(logp_a - torch.as_tensor(concatenated_logp_as, dtype=torch.float32))
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * torch.as_tensor(gae_values,
                                                                                                  dtype=torch.float32)

        loss_pi = -(torch.min(ratio * torch.as_tensor(gae_values, dtype=torch.float32), clip_adv)).mean()
        targets_n = torch.as_tensor((q_values - np.mean(q_values)) / (np.std(q_values) + 1e-8), dtype=torch.float32)
        predicted_n = self.policy_agent.get_value(torch.as_tensor(obs, dtype=torch.float32), adapted_params)
        loss_v = ((predicted_n - targets_n) ** 2).mean()

        return loss_pi + self.c1 * loss_v

    def discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def calculate_advantages(self, obs, q_values, params):
        b_n_unnormalized = self.policy_agent.get_value(torch.as_tensor(obs, dtype=torch.float32), params)
        b_n = b_n_unnormalized.detach().numpy() * np.std(q_values) + np.mean(q_values)
        adv_n = q_values - b_n

        # Normalize the resulting advantages
        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def calculate_gae(self, rewards, obs, params, lam):

        values = self.policy_agent.get_value(torch.as_tensor(obs, dtype=torch.float32), params).detach().numpy()

        advantages = []
        advantage = 0.0
        next_value = 0.0

        for r, v in zip(reversed(rewards), reversed(values)):
            td_error = r + next_value * self.gamma - v
            advantage = td_error + advantage * self.gamma * lam
            next_value = v
            advantages.insert(0, advantage)

        advantages = torch.tensor(advantages)

        if self.standardize_advantages:
            advantages = (advantages - advantages.mean()) / advantages.std()

        return np.array(advantages, dtype=np.float32)