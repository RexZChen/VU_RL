"""
Utils functions

Author: Zirong Chen
"""
import torch
import torch.nn as nn
import pandas as pd


# env: BipedalWalker-v3
# observation shape: (24,)
# action space is: Continuous
# action shape: (4,)
# action lower bound: [-1. -1. -1. -1.]
# action higher bound: [1. 1. 1. 1.]

def layer_init(layer, scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def weights_init_normal(layers, mean, std):
    for layer in layers:
        if isinstance(layer, nn.Linear):
            layer.weight.data.normal_(mean, std)


def mlp(sizes, activation, output_activation=nn.Identity, initialize=True):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    if initialize:
        weights_init_normal(layers, 0.0, 0.1)
    return nn.Sequential(*layers)


def log2file(a_list, algo, env):
    file_name = algo + "_" + env
    data = []

    for epoch, pi_loss, vf_loss, avg_reward, avg_steps, num_of_traj in a_list:
        data.append([epoch, pi_loss, vf_loss, avg_reward, avg_steps, num_of_traj])

    df = pd.DataFrame(data, columns=["epoch", "pi_loss", "vf_loss", "avg_reward", "avg_steps", "num_of_traj"])

    df.to_csv(file_name)
