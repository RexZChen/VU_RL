from reinforce_chenz51 import ReinforceAgent
from ddpg_chenz51 import DDPGAgent
from ddpg_chenz51_new import DDPGAgent_new
from replay_buffer import *
import torch
import torch.nn as nn
import gym
from utils import *

DEVICE = torch.device("cpu")

env = gym.make("MountainCarContinuous-v0")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]
params = dict()
params["device"] = DEVICE
params["obs_dim"] = obs_dim
params["act_dim"] = act_dim
params["gamma"] = 0.99
params["lam"] = 0.97
params["train_v_iters"] = 80
params["pi_lr"] = 3e-4
params["v_lr"] = 1e-3
params["steps_per_epoch"] = 4000
params["max_ep_len"] = 1000
params["replay_buffer"] = OnPolicyBuffer(obs_dim=params["obs_dim"],
                                         act_dim=params["act_dim"],
                                         size=params["steps_per_epoch"],
                                         gamma=params["gamma"],
                                         lam=params["lam"])
# agent = ReinforceAgent(params)
#
# torch.save(agent.actor_critic.state_dict(), "test.h5")
#
# new_agent = ReinforceAgent(params)
# new_agent.actor_critic.load_state_dict(torch.load("test.h5"))
#
# print(new_agent)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

act_low = env.action_space.low[0]
act_high = env.action_space.high[0]
new_params = dict()
new_params["obs_dim"] = obs_dim
new_params["act_dim"] = act_dim
new_params["hidden_sizes"] = [64] * 2
new_params["batch_size"] = 64
new_params["replay_buffer"] = DDPGBuffer(capacity=1000)
new_params["act_high"] = env.action_space.high[0]
new_params["act_low"] = env.action_space.low[0]
new_params["device"] = DEVICE
new_params["gamma"] = 0.99
new_params["polyak"] = 0.995
new_params["noise_scale"] = 0.15
new_params["pi_lr"] = 5e-4
new_params["q_lr"] = 5e-4

new_DDPG_agent = DDPGAgent_new(new_params)
o = env.reset()
print(new_DDPG_agent.choose_action(torch.tensor(o, dtype=torch.float32)))