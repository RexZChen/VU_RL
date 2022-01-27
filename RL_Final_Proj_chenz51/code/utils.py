"""
This class provides utility function for RL

Author: Zirong Chen

Reference: UC Berkeley CS285 homework framework
"""
import numpy as np
import torch
import pandas as pd


def sample_trajectory(env, policy_agent, params, max_path_length, limit_path_length=False, render=False):

    """
    :param env: environment to interact with
    :param policy: current agent's policy
    :param max_path_length: max length of trajectory
    :param limit_path_length: whether to set max_path_length or not
    :param render: whether to render the environment or not
    :return: one sampled trajectory
    """
    observation = env.reset()
    observations, actions, rewards, logp_as, terminals = [], [], [], [], []
    steps = 0
    while True:
        if render:
            env.render()
        observations.append(observation.copy())
        action, _, logp_a = policy_agent.get_act(torch.as_tensor(observation, dtype=torch.float32), params)
        logp_as.append(logp_a)
        actions.append(action)
        observation, reward, done, _ = env.step(action)
        steps += 1
        rewards.append(reward)
        rollout_done = (steps == max_path_length or done) if limit_path_length else done
        terminals.append(rollout_done)
        if rollout_done:
            break
    return Path(observations, actions, rewards, logp_as, terminals)


def sample_n_trajectories(env, policy_agent, params, num_trajectory, max_path_length, limit_path_length=False, render=False):
    paths = []
    for _ in range(num_trajectory):
        paths.append(sample_trajectory(env, policy_agent, params, max_path_length, limit_path_length, render))
    return paths


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])  # one whole list
    unconcatenated_rewards = [path["reward"] for path in paths]  # list of list
    concatenated_logp_as = np.concatenate([path["logp_a"] for path in paths])
    terminals = np.concatenate([path["terminals"] for path in paths])
    return observations, actions, concatenated_rewards, unconcatenated_rewards, concatenated_logp_as, terminals


def Path(obs, acs, rewards, logp_as, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    return {"observation": np.array(obs, dtype=np.float32),  # (n, obs_dim)
            "reward": np.array(rewards, dtype=np.float32),
            "action": np.array(acs, dtype=np.float32),  # (n, act_dim)
            "logp_a": np.array(logp_as, dtype=np.float32),  # (n)
            "terminals": np.array(terminals, dtype=np.float32)}


def compute_total_returns(meta_paths):
    return [path["reward"].sum() for path in meta_paths]


def log2file(batch_step_list, batch_avg_reward_list, algo):
    """log to file"""

    assert len(batch_step_list) == len(batch_avg_reward_list)

    file_name = algo

    rewards = []

    for i, reward in enumerate(batch_avg_reward_list):
        num_step = batch_step_list[i]
        rewards += [reward] * num_step

    column_name = 'avg reward'
    df = pd.DataFrame(rewards, columns=[column_name])
    df.to_csv(file_name)
