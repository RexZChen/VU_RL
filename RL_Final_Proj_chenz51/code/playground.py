import gym
from maml_env import HalfCheetahDirecBulletEnv

env = HalfCheetahDirecBulletEnv()

discrete = isinstance(env.action_space, gym.spaces.Discrete)
print(discrete)