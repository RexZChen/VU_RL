"""
Utils functions for further references

Author: Zirong Chen
"""
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import pandas as pd


def weights_init_normal(layers, mean, std):
    for layer in layers:
        if isinstance(layer, nn.Linear):
            layer.weight.data.normal_(mean, std)


def mlp(sizes, activation, output_activation=nn.Identity, initialize=True):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    if initialize:
        weights_init_normal(layers, 0.0, 0.1)
    return nn.Sequential(*layers)


def mlp_ddpg(sizes, activation, output_activation=nn.Identity, initialize=True):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    if initialize:
        weights_init_normal(layers, 0.0, 0.1)
    return nn.Sequential(*layers)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def log2file_vpg(a_list, algo, env):
    """
    :param a_list: [epoch, pi_loss, v_loss, avg_return]
    :param algo:
    :param env:
    :return:
    """

    file_name = algo + "_" + env
    data = []
    for epoch, pi_loss, vf_loss, avg_return in a_list:
        data.append([epoch, pi_loss, vf_loss, avg_return])
    df = pd.DataFrame(data, columns=['Epochs', 'pi_loss', 'vf_loss', 'avg_returns'])

    df.to_csv(file_name)


def log2file_ddpg(a_list, algo, env):
    """
    :param a_list: [traj, pi_loss, v_loss, avg_return, length]
    :param algo:
    :param env:
    :return:
    """

    file_name = algo + "_" + env
    data = []
    for epoch, pi_loss, vf_loss, avg_return, length in a_list:
        data.append([epoch, pi_loss, vf_loss, avg_return, length])
    df = pd.DataFrame(data, columns=['traj', 'pi_loss', 'q_loss', 'avg_returns', 'length'])

    df.to_csv(file_name)