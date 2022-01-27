from ppo_chenz51 import PPOAgent
from replay_buffer import *
import gym
import torch
from utils import *

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
    ppo_agent.actor_critic.load_state_dict(torch.load("PPO_BipedalWalker.h5"))

    # print(ppo_agent.actor_critic)
    for _ in range(10):
        o = env.reset()
        done = False
        while not done:
            act, logp_a, v = ppo_agent.choose_action(torch.tensor(o[None, :], dtype=torch.float32).to(DEVICE))
            next_o, r, d, _ = env.step(act)
            env.render()
            done = d
            o = next_o

    env.close()


if __name__ == "__main__":
    torch.manual_seed(157)
    np.random.seed(157)

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_layers', type=int, default=2)  # 2
    parser.add_argument('--hidden_size', type=int, default=64)  # 256
    parser.add_argument("--epoch", default=5000, type=int)
    parser.add_argument("--steps_per_epoch", default=2500, type=int)
    parser.add_argument('--train_v_iters', '-v_it', type=int, default=10)
    parser.add_argument('--train_pi_iters', '-p_it', type=int, default=1)
    parser.add_argument('--pi_lr', type=float, default=2.5e-4)
    parser.add_argument('--vf_lr', type=float, default=2.5e-4)
    parser.add_argument('--entropy_coef', '-ef', type=float, default=0.001)
    args = parser.parse_args()

    ppo(args)
