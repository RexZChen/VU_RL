"""
Train sessions
reinforce: REINFORCE alg
ddpg: DDPG traj-wise control
ddpg_steps: DDPG step-wise control
ddpg_new: DDPG traj-wise control but with newly designed buffer

Author: Zirong Chen
"""
from reinforce_chenz51 import ReinforceAgent
from ddpg_chenz51 import DDPGAgent
from ddpg_chenz51_new import DDPGAgent_new
from replay_buffer import *
import torch
import torch.nn as nn
import gym
from utils import *


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
MAX_LEN_PER_EPISODE = 500

env = gym.make("LunarLanderContinuous-v2")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

act_low = env.action_space.low[0]
act_high = env.action_space.high[0]

print("obs dim: ", obs_dim)
print("act_dim: {}".format(act_dim))


def reinforce(args):
    reinforce_params = dict()
    reinforce_params["device"] = DEVICE
    reinforce_params["obs_dim"] = obs_dim
    reinforce_params["act_dim"] = act_dim
    reinforce_params["gamma"] = args.gamma
    reinforce_params["lam"] = args.lam
    reinforce_params["train_v_iters"] = args.vfiters
    reinforce_params["pi_lr"] = args.pi_lr
    reinforce_params["v_lr"] = args.vf_lr
    reinforce_params["steps_per_epoch"] = args.steps_per_epoch
    reinforce_params["max_ep_len"] = args.max_ep_len
    reinforce_params["replay_buffer"] = OnPolicyBuffer(obs_dim=reinforce_params["obs_dim"],
                                                       act_dim=reinforce_params["act_dim"],
                                                       size=reinforce_params["steps_per_epoch"],
                                                       gamma=reinforce_params["gamma"],
                                                       lam=reinforce_params["lam"])

    agent = ReinforceAgent(reinforce_params)
    var_counts = tuple(count_vars(module) for module in [agent.actor_critic.pi, agent.actor_critic.v])
    print('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    o, ep_ret, ep_len = env.reset(), 0, 0
    log_file = []
    epoch = 0
    while True and epoch < args.epochs:
        rewards_to_check = []
        if epoch > 100:
            for _, _, _, avg_rew in log_file[-100:]:
                rewards_to_check.append(avg_rew)

            if min(rewards_to_check) > 90:
                break
        traj_reward_list = []
        reward_list = []
        for t in range(args.steps_per_epoch):
            a, v, logp = agent.actor_critic.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = env.step(a)
            reward_list.append(r)
            ep_ret += r
            ep_len += 1

            agent.replay_buffer.store(o, a, r, v, logp)

            o = next_o

            timeout = ep_len == reinforce_params["max_ep_len"]
            terminal = d or timeout
            epoch_ended = t == args.steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    pass
                    # print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)

                if timeout or epoch_ended:
                    _, v, _ = agent.actor_critic.step(torch.as_tensor(o, dtype=torch.float32))

                else:
                    v = 0

                agent.replay_buffer.finish_path(v)
                if terminal:
                    pass

                traj_reward_list.append(reward_list)
                reward_list = []
                o, ep_ret, ep_len = env.reset(), 0, 0

            if len(reward_list) > 0:
                traj_reward_list.append(reward_list)

        pi_loss, v_loss = agent.train()
        ep_reward_list = [sum(traj) for traj in traj_reward_list]
        total_rewards = sum(ep_reward_list)
        average_reward = total_rewards / len(traj_reward_list)
        print('[%d] => pi_loss: %.10f, v_loss: %.10f, avg returns: %.10f' % (
            epoch + 1, pi_loss, v_loss, average_reward))

        log_file.append([epoch + 1, pi_loss, v_loss, average_reward])
        epoch += 1

    log2file_vpg(log_file, "VPG", "LunarLander")
    torch.save(agent.actor_critic.state_dict(), "VPG_LunarLander.h5")
    env.close()


def ddpg(args):
    ddpg_params = dict()
    ddpg_params["device"] = DEVICE
    ddpg_params["act_dim"] = act_dim
    ddpg_params["obs_dim"] = obs_dim
    ddpg_params["act_high"] = act_high
    ddpg_params["act_low"] = act_low
    ddpg_params["replay_buffer"] = OffPolicyReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=args.replay_size)
    ddpg_params["batch_size"] = args.batch_size
    ddpg_params["gamma"] = args.gamma
    ddpg_params["polyak"] = args.polyak
    ddpg_params["noise_scale"] = args.act_noise
    ddpg_params["pi_lr"] = args.pi_lr
    ddpg_params["q_lr"] = args.q_lr

    agent = DDPGAgent(ddpg_params)
    # print(agent)
    var_counts = tuple(count_vars(module) for module in [agent.actor_critic.pi, agent.actor_critic.q])
    print('\nNumber of parameters: \t pi: %d, \t q: %d\n' % var_counts)

    start_train_step = args.batch_size
    # update_frequency = args.update_frequency

    reward_list = []
    step_list = []
    t = 0

    for i in range(args.num_trajectories):
        obs = env.reset()
        done = False
        traj_reward_list = []
        step = 0

        while not done and (step < MAX_LEN_PER_EPISODE if args.max_length_limit else True):

            act = agent.choose_action(torch.tensor(obs, dtype=torch.float32))
            next_obs, reward, done, _ = env.step(act)
            traj_reward_list.append(reward)
            # store(self, obs, act, rew, next_obs, done):
            agent.replay_buffer.store(obs, act, reward, next_obs, done)
            obs = next_obs

            t += 1
            step += 1

            if agent.replay_buffer.size < start_train_step:
                continue

            loss_q, loss_pi = agent.train()

        print("[{}] trajectory => len: {}, reward: sum: {}".format(i+1, len(traj_reward_list), sum(traj_reward_list)))
        reward_list.append(sum(traj_reward_list))
        step_list.append(len(traj_reward_list))

    env.close()


def ddpg_steps(args):
    steps_per_epoch = 1000
    start_steps = 1000
    update_after = 1000
    update_every = 20
    max_ep_len = 500
    epochs = 5000

    ddpg_step_params = dict()
    ddpg_step_params["device"] = DEVICE
    ddpg_step_params["act_dim"] = act_dim
    ddpg_step_params["obs_dim"] = obs_dim
    ddpg_step_params["act_high"] = act_high
    ddpg_step_params["act_low"] = act_low
    ddpg_step_params["replay_buffer"] = OffPolicyReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=args.replay_size)
    ddpg_step_params["batch_size"] = args.batch_size
    ddpg_step_params["gamma"] = args.gamma
    ddpg_step_params["polyak"] = args.polyak
    ddpg_step_params["noise_scale"] = args.act_noise
    ddpg_step_params["pi_lr"] = args.pi_lr
    ddpg_step_params["q_lr"] = args.q_lr

    agent = DDPGAgent(ddpg_step_params)
    # print(agent)
    var_counts = tuple(count_vars(module) for module in [agent.actor_critic.pi, agent.actor_critic.q])
    print('\nNumber of parameters: \t pi: %d, \t q: %d\n' % var_counts)

    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0

    traj_reward_list = []
    reward_list = []
    loss_info_q = []
    loss_info_pi = []
    log_file = []

    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = agent.choose_action(torch.tensor(o, dtype=torch.float32))

        else:
            a = env.action_space.sample()

        # step the env
        o2, r, d, _ = env.step(a)
        reward_list.append(r)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        agent.replay_buffer.store(o, a, r, o2, d)

        o = o2
        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0
            traj_reward_list.append(reward_list)
            reward_list = []

        # Update handling
        if t >= update_after and t % update_every == 0:
            batch_loss_pi = 0.
            batch_loss_q = 0.
            for _ in range(update_every):
                loss_q, loss_pi = agent.train()
                batch_loss_q += loss_q.item()
                batch_loss_pi += loss_pi.item()
                loss_info_q.append(batch_loss_q / update_every)
                loss_info_pi.append(batch_loss_pi / update_every)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            ep_reward_list = [sum(traj) for traj in traj_reward_list]
            total_rewards = sum(ep_reward_list)
            average_reward = total_rewards / len(traj_reward_list)
            ep_len_list = [len(traj) for traj in traj_reward_list]
            total_steps = sum(ep_len_list)
            average_steps = total_steps / len(traj_reward_list)
            pi_loss = np.mean(loss_info_pi)
            q_loss = np.mean(loss_info_q)
            print('[%d] => average reward: %.5f, avg steps: %.2f, num of trajs: %d, pi_loss: %.10f, q_loss: %.10f' %
                  (epoch, average_reward, average_steps, len(traj_reward_list), pi_loss, q_loss))
            traj_reward_list = []

            log_file.append([epoch, pi_loss, q_loss, average_reward, average_steps])

    log2file_ddpg(log_file, "DDPG", "LunarLander_new")
    torch.save(agent.actor_critic.state_dict(), "DDPG_LunarLander_new.h5")
    env.close()


def ddpg_new(args):
    new_params = dict()
    new_params["obs_dim"] = obs_dim
    new_params["act_dim"] = act_dim
    new_params["hidden_sizes"] = [args.hidden_sizes] * args.num_layers
    new_params["batch_size"] = 64
    new_params["replay_buffer"] = DDPGBuffer(capacity=2000)
    new_params["act_high"] = env.action_space.high[0]
    new_params["act_low"] = env.action_space.low[0]
    new_params["device"] = DEVICE
    new_params["gamma"] = 0.99
    new_params["polyak"] = 0.995
    new_params["noise_scale"] = 0.15
    new_params["pi_lr"] = 1e-3
    new_params["q_lr"] = 1e-3

    new_DDPG_agent = DDPGAgent_new(new_params)

    start_train_step = args.batch_size  # num of steps agent starts training
    # update_frequency = args.update_frequency  # num of every steps target q network get updated

    reward_list = []
    step_list = []
    t = 0
    log_file = []
    i = 0
    while True and i < args.num_trajectories:
        rewards_to_check = []
        # log_file.append([i + 1, avg_loss_pi, avg_loss_q, sum(traj_reward_list), len(traj_reward_list)])
        if i > 100:
            for _, _, _, avg_rew, _ in log_file[-100:]:
                rewards_to_check.append(avg_rew)

            if min(rewards_to_check) > 200:
                break
        NUM_EPOCH = 500
        STEPS_PER_EPOCH = 1000
        NUM_TRAJECTORIES = 200000
        MAX_LENGTH_PER_EPISODE = 500
        START_STEPS = 1000

        obs = env.reset()
        done = False
        traj_reward_list = []
        step = 0
        batch_loss_sum_pi = 0.
        batch_loss_sum_q = 0.
        while not done and (step < MAX_LENGTH_PER_EPISODE if args.max_length_limit else True):
            act = new_DDPG_agent.choose_action(torch.tensor(obs, dtype=torch.float32))
            next_obs, reward, done, _ = env.step(act)
            traj_reward_list.append(reward)
            new_DDPG_agent.replay_buffer.add_transitions((obs, act, reward, done, next_obs))
            obs = next_obs
            t += 1
            step += 1

            if new_DDPG_agent.replay_buffer.size() < start_train_step:
                continue

            loss_q, loss_pi = new_DDPG_agent.train()
            batch_loss_sum_pi += loss_pi
            batch_loss_sum_q += loss_q

        avg_loss_q = np.mean(batch_loss_sum_q)
        avg_loss_pi = np.mean(batch_loss_sum_pi)
        print("[{}] trajectory => len: {}, reward sum: {}, loss_pi: {}, loss_q: {}".format(i+1, len(traj_reward_list), sum(traj_reward_list), avg_loss_pi, avg_loss_q))
        reward_list.append(sum(traj_reward_list))
        step_list.append(len(traj_reward_list))

        log_file.append([i + 1, avg_loss_pi, avg_loss_q, sum(traj_reward_list), len(traj_reward_list)])

        i += 1

    # for i in range(args.num_trajectories):
    #     NUM_EPOCH = 500
    #     STEPS_PER_EPOCH = 1000
    #     NUM_TRAJECTORIES = 200000
    #     MAX_LENGTH_PER_EPISODE = 500
    #     START_STEPS = 1000
    #
    #     obs = env.reset()
    #     done = False
    #     traj_reward_list = []
    #     step = 0
    #     batch_loss_sum_pi = 0.
    #     batch_loss_sum_q = 0.
    #     while not done and (step < MAX_LENGTH_PER_EPISODE if args.max_length_limit else True):
    #         act = new_DDPG_agent.choose_action(torch.tensor(obs, dtype=torch.float32))
    #         next_obs, reward, done, _ = env.step(act)
    #         traj_reward_list.append(reward)
    #         new_DDPG_agent.replay_buffer.add_transitions((obs, act, reward, done, next_obs))
    #         obs = next_obs
    #         t += 1
    #         step += 1
    #
    #         if new_DDPG_agent.replay_buffer.size() < start_train_step:
    #             continue
    #
    #         loss_q, loss_pi = new_DDPG_agent.train()
    #         batch_loss_sum_pi += loss_pi
    #         batch_loss_sum_q += loss_q
    #
    #     avg_loss_q = np.mean(batch_loss_sum_q)
    #     avg_loss_pi = np.mean(batch_loss_sum_pi)
    #     print("[{}] trajectory => len: {}, reward sum: {}, loss_pi: {}, loss_q: {}".format(i+1, len(traj_reward_list), sum(traj_reward_list), avg_loss_pi, avg_loss_q))
    #     reward_list.append(sum(traj_reward_list))
    #     step_list.append(len(traj_reward_list))
    #
    #     log_file.append([i + 1, avg_loss_pi, avg_loss_q, sum(traj_reward_list), len(traj_reward_list)])

    log2file_ddpg(log_file, "DDPG", "LunarLander")
    torch.save(new_DDPG_agent.actor.state_dict(), "DDPG_LunarLander.h5")
    env.close()


if __name__ == "__main__":
    torch.manual_seed(157)
    np.random.seed(157)

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", "-a", required=True, help="Please enter DDPG or VPG")
    parser.add_argument("--hidden_sizes", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--pi_lr", type=float, default=0.001)
    parser.add_argument("--vf_lr", type=float, default=1e-3)
    parser.add_argument("--q_lr", type=float, default=1e-3)
    parser.add_argument("--vfiters", type=int, default=80)
    # parser.add_argument("--piiters", type=int, default=1)
    parser.add_argument("--max_ep_len", type=int, default=1000)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--replay_size", type=int, default=10000)
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--act_noise", type=float, default=0.1)
    parser.add_argument("--start_steps", type=int, default=500)
    parser.add_argument("--update_after", type=int, default=500)
    parser.add_argument("--update_every", type=int, default=100)
    parser.add_argument('--update_frequency', '-uf', type=int, default=150)
    parser.add_argument('--num_trajectories', '-nt', type=int, default=10000)
    parser.add_argument("--max_length_limit", default=False)
    args = parser.parse_args()

    # print(args)

    if args.algorithm == "VPG" or args.algorithm == "reinforce":
        print("TRAINING VPG....")
        reinforce(args)

    if args.algorithm == "DDPG" or args.algorithm == "ddpg":
        print("TRAINING DDPG...")
        ddpg_steps(args)
