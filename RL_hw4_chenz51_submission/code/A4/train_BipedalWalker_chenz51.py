import torch
import torch.nn as nn
import numpy as np
from utils import *
from ppo_chenz51 import PPOAgent
from replay_buffer import *
import gym
import pybullet_envs

DEVICE = torch.device("cpu")
env = gym.make("BipedalWalker-v3")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

act_low = env.action_space.low[0]
act_high = env.action_space.high[0]

print("obs dim: ", obs_dim)
print("act_dim: {}".format(act_dim))


def ppo(args):
    ppo_params = dict()
    ppo_params["device"] = DEVICE
    ppo_params["act_dim"] = act_dim
    ppo_params["obs_dim"] = obs_dim
    ppo_params["act_high"] = act_high
    ppo_params["act_low"] = act_low
    ppo_params["hidden_sizes"] = [args.hidden_size] * args.num_layers
    ppo_params["gamma"] = 0.99
    ppo_params["lambda"] = 0.95
    ppo_params["clip_ratio"] = 0.2
    ppo_params["entropy_coef"] = args.entropy_coef
    ppo_params["train_v_iters"] = args.train_v_iters
    ppo_params["train_pi_iters"] = args.train_pi_iters
    ppo_params["pi_lr"] = args.pi_lr
    ppo_params["vf_lr"] = args.vf_lr
    ppo_params["replay_buffer"] = OnPolicyBuffer(ppo_params['gamma'],
                                                 ppo_params['lambda'],
                                                 obs_dim,
                                                 act_dim,
                                                 args.steps_per_epoch)

    ppo_agent = PPOAgent(ppo_params)
    batch_avg_reward_list = []
    batch_step_list = []
    obs = env.reset()
    done = False
    v = 0
    log_file = []
    for epoch in range(args.epoch):
        traj_reward_list = []
        reward_list = []
        for i in range(args.steps_per_epoch):
            act, logp_a, v = ppo_agent.choose_action(torch.tensor(obs[None, :], dtype=torch.float32).to(DEVICE))
            next_obs, reward, done, _ = env.step(act)
            reward_list.append(reward)
            ppo_agent.replay_buffer.add_transition(obs, act, reward, v, logp_a)
            if done:
                ppo_agent.replay_buffer.finish_episode(0)
                next_obs = env.reset()
                traj_reward_list.append(reward_list)
                # print("trajectory reward: sum: {}, {}".format(sum(reward_list), reward_list))
                reward_list = []
            obs = next_obs

        if not done:
            ppo_agent.replay_buffer.finish_episode(v)
        if len(reward_list) > 0:
            traj_reward_list.append(reward_list)

        loss_pi, loss_v = ppo_agent.train()

        # log statistics
        ep_reward_list = [sum(traj) for traj in traj_reward_list]
        total_rewards = sum(ep_reward_list)
        average_reward = total_rewards / len(traj_reward_list)
        batch_avg_reward_list.append(average_reward)
        ep_len_list = [len(traj) for traj in traj_reward_list]
        total_steps = sum(ep_len_list)
        batch_step_list.append(total_steps)
        average_steps = total_steps / len(traj_reward_list)

        print('[%d] pi loss: %.3f; vf loss: %.3f => average reward: %.5f, average steps: %.2f, num of trajs: %d' %
              (epoch + 1, loss_pi, loss_v, average_reward, average_steps, len(traj_reward_list)))

        log_file.append([epoch + 1, loss_pi, loss_v, average_reward, average_steps, len(traj_reward_list)])

    log2file(log_file, "PPO", "BipedalWalker_256_2000")
    torch.save(ppo_agent.actor_critic.state_dict(), "PPO_BipedalWalker_256_2000.h5")
    env.close()


if __name__ == "__main__":
    torch.manual_seed(157)
    np.random.seed(157)

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_layers', type=int, default=2)  # 2
    parser.add_argument('--hidden_size', type=int, default=256)  # 256
    parser.add_argument("--epoch", default=2000, type=int)  # 10000
    parser.add_argument("--steps_per_epoch", default=2500, type=int)
    parser.add_argument('--train_v_iters', '-v_it', type=int, default=80)  # 50
    parser.add_argument('--train_pi_iters', '-p_it', type=int, default=20)  # 1
    parser.add_argument('--pi_lr', type=float, default=3e-4)  # 2.5e-4
    parser.add_argument('--vf_lr', type=float, default=1e-3)  # 2.5e-4
    parser.add_argument('--entropy_coef', '-ef', type=float, default=0.001)
    args = parser.parse_args()

    ppo(args)
