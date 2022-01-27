from maml_env import HalfCheetahDirecBulletEnv
import random
import gym
import torch
import numpy as np
from utils import sample_n_trajectories, compute_total_returns
from policy_agent import PolicyAgent
from agent import MetaLearnerAgent
from collections import OrderedDict, defaultdict
from replay_buffer import OnPolicyBuffer
import os


class Tasks:
    def __init__(self, *task_configs):
        self.tasks = [i for i in task_configs]

    def sample_tasks(self, batch_size):
        return random.choices(self.tasks, k=batch_size)


def main(args):

    torch.manual_seed(2021)
    random.seed(2021)
    np.random.seed(2021)

    tasks = Tasks(("Forward", True), ("Backward", False))

    env = HalfCheetahDirecBulletEnv()

    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n if discrete else env.action_space.shape[0]

    hidden_sizes = [args.hidden_size] * args.num_layers

    policy_agent = PolicyAgent(obs_dim, act_dim, hidden_sizes)

    meta_learner = MetaLearnerAgent(policy_agent,
                                    args.discount,
                                    args.lam,
                                    args.clip_ratio,
                                    args.c1,
                                    args.meta_lr)

    policy_list = []  # [(policy, average return)]
    average_return_list = []

    env.close()

    # Outer loop
    for meta_iter in range(args.meta_iteration):

        adapted_params_list = []  # save parameters that adapted to each task
        meta_paths_list = []
        task_return_dict = defaultdict(list)
        total_return_list = []

        for task_config in tasks.sample_tasks(args.meta_batch_size):
            # Inner loop
            task_name, env_args = task_config[0], task_config[1:]
            env = HalfCheetahDirecBulletEnv(*env_args)

            adapted_params = OrderedDict(policy_agent.named_parameters())

            # Adaptation
            for i in range(args.num_adapt_steps):
                # Collect trajectories
                buffer = OnPolicyBuffer()
                paths = sample_n_trajectories(env, policy_agent, adapted_params, args.num_trajectory,
                                              args.horizon, True)
                env.render()
                buffer.add_rollouts(paths)
                alpha = args.alpha if i == 0 else 0.5 * args.alpha
                adapted_params = meta_learner.adapt(alpha, buffer, adapted_params)
            adapted_params_list.append(adapted_params)
            # Run adapted policy (collect new trajectories using new policy)
            meta_paths = sample_n_trajectories(env, policy_agent, adapted_params, args.num_trajectory, args.horizon, True)
            meta_paths_list.append(meta_paths)

            total_returns = compute_total_returns(meta_paths)
            print("total returns: ", total_returns)
            task_return_dict[task_name] += total_returns
            total_return_list += total_returns
            env.close()

        # Meta Optimization
        loss = meta_learner.step(adapted_params_list, meta_paths_list)
        print("{} iteration".format(meta_iter))
        print("MetaLoss: ", loss)
        print("Validation AverageReturn: ", np.mean(total_return_list))
        average_return_list.append(np.mean(total_return_list))
        for task_name, return_list in task_return_dict.items():
            print("Return/{}".format(task_name), np.mean(return_list))

        # save policy and its validation average return
        dir_name = "./2021_saved_policy_{}/meta_{}".format(args.num_adapt_steps, meta_iter)
        os.makedirs(dir_name)
        torch.save(meta_learner.policy_agent, dir_name + "/model.pt")
        policy_list.append((dir_name, np.mean(total_return_list)))

    # save the average return list
    return_dir_name = "Res/2021_average_return"
    os.makedirs(return_dir_name, exist_ok=True)
    np.savetxt(return_dir_name + "/{}_adapt.csv".format(args.num_adapt_steps),
               np.array(average_return_list))

    # pick the policy with highest validation average return,
    #  sample trajectories, adapt and compute the average return
    dir_name, highest_return = sorted(policy_list, key=lambda x: x[1], reverse=True)[0]
    print("{} dir has highest return: {}".format(dir_name, highest_return))
    best_policy_agent = torch.load(dir_name + "/model.pt")

    test_meta_learner = MetaLearnerAgent(best_policy_agent,
                                         args.discount,
                                         args.lam,
                                         args.clip_ratio,
                                         args.c1,
                                         args.meta_lr)

    test_task_return_dict = defaultdict(list)
    test_total_return_list = []

    # test
    for task_config in tasks.sample_tasks(args.meta_batch_size):
        # Inner loop
        task_name, env_args = task_config[0], task_config[1:]
        env = HalfCheetahDirecBulletEnv(*env_args)

        adapted_params = OrderedDict(test_meta_learner.policy_agent.named_parameters())

        # Adaptation
        for i in range(args.num_adapt_steps):
            # Collect trajectories
            buffer = OnPolicyBuffer()
            paths = sample_n_trajectories(env, test_meta_learner.policy_agent, adapted_params, args.num_trajectory,
                                          args.horizon,
                                          True)
            buffer.add_rollouts(paths)
            alpha = args.alpha if i == 0 else 0.5 * args.alpha
            adapted_params = test_meta_learner.adapt(alpha, buffer, adapted_params)
        # Run adapted policy (collect new trajectories using new policy)
        meta_paths = sample_n_trajectories(env, test_meta_learner.policy_agent, adapted_params, args.num_trajectory,
                                           args.horizon, True)

        test_total_returns = compute_total_returns(meta_paths)
        print("test total returns: ", test_total_returns)
        test_task_return_dict[task_name] += test_total_returns
        test_total_return_list += test_total_returns
        env.close()

    print("Test AverageReturn: ", np.mean(test_total_return_list))
    for task_name, return_list in test_task_return_dict.items():
        print("Test Return/{}".format(task_name), np.mean(return_list))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--meta_iteration", default=150, type=int)
    parser.add_argument("--meta_batch_size", default=40, type=int)
    parser.add_argument("--horizon", "-H", default=200, type=int)
    parser.add_argument("--num_adapt_steps", default=0, type=int)
    parser.add_argument("--alpha", default=0.1, type=float)  # inner learning rate
    parser.add_argument('--discount', type=float, default=0.99)  # gamma
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument("--num_trajectory", "-K", default=20, type=int)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--c1', type=float, default=0.5)
    parser.add_argument('--meta_lr', type=float, default=1e-4)

    args = parser.parse_args()

    main(args)
