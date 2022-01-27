import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from collections import OrderedDict


class PolicyAgent(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=torch.tanh):
        super(PolicyAgent, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.num_layers = len(hidden_sizes) + 1
        self._build_mlp([obs_dim] + hidden_sizes + [act_dim], "pi")
        self._build_mlp([obs_dim] + hidden_sizes + [1], "v")
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=True)

    def _build_mlp(self, layer_sizes, namespace):
        num_layers = len(layer_sizes)
        for i in range(1, num_layers):
            self.add_module("{}_layer_{}".format(namespace, i), nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

    def _model_forward(self, obs, namespace, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = obs
        for i in range(1, self.num_layers):
            output = F.linear(output,
                              params["{}_layer_{}.weight".format(namespace, i)],
                              params["{}_layer_{}.bias".format(namespace, i)])
            output = self.activation(output)
        output = F.linear(output,
                          params["{}_layer_{}.weight".format(namespace, self.num_layers)],
                          params["{}_layer_{}.bias".format(namespace, self.num_layers)])

        return output

    def get_value(self, obs, params=None):
        v = self._model_forward(obs, "v", params)
        return torch.squeeze(v, dim=-1)

    def get_pi(self, obs, act, params=None):
        mu = self._model_forward(obs, "pi", params)
        if params is None:
            pi = Normal(mu, torch.exp(self.log_std))
        else:
            pi = Normal(mu, torch.exp(params['log_std']))
        log_p = pi.log_prob(act).sum(axis=-1)
        return pi, log_p

    def get_act(self, obs, params=None):
        with torch.no_grad():
            v = self.get_value(obs, params)
            mu = self._model_forward(obs, "pi", params)
            if params is None:
                pi = Normal(mu, torch.exp(self.log_std))
            else:
                pi = Normal(mu, torch.exp(params["log_std"]))

            act = pi.sample()
            log_p = pi.log_prob(act).sum(axis=-1)

        return act.numpy(), v.numpy(), log_p.numpy()

    def update_params(self, grads, step_size=0.5):
        updated_params = OrderedDict()
        for (name, param), grad in zip(self.named_parameters(), grads):
            updated_params[name] = param - step_size * grad

        return updated_params