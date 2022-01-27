import numpy as np

from utils import *


class OnPolicyBuffer:
    def __init__(self):
        self.paths = []
        self.obs = None
        self.acts = None
        self.concatenated_rews = None
        self.unconcatenated_rews = []
        self.concatenated_logp_as = None
        self.terminals = None

    def add_rollouts(self, paths):
        self.paths += paths

        # convert new rollouts into their component arrays, and append them onto our arrays
        observations, actions, concatenated_rews, unconcatenated_rews, concatenated_logp_as, terminals \
            = convert_listofrollouts(paths)
        self.obs = observations if self.obs is None else np.concatenate([self.obs, observations])  # (n, obs_dim)
        self.acts = actions if self.acts is None else np.concatenate([self.acts, actions])  # (n, act_dim)
        self.concatenated_rews = concatenated_rews if self.concatenated_rews is None \
            else np.concatenate([self.concatenated_rews, concatenated_rews])
        if isinstance(unconcatenated_rews, list):
            self.unconcatenated_rews += unconcatenated_rews
        else:
            self.unconcatenated_rews.append(unconcatenated_rews)
        self.concatenated_logp_as = concatenated_logp_as if self.concatenated_logp_as is None \
            else np.concatenate([self.concatenated_logp_as, concatenated_logp_as])
        self.terminals = terminals if self.terminals is None else np.concatenate([self.terminals, terminals])

    def evict(self):
        self.paths = []
        self.obs = None
        self.acts = None
        self.concatenated_rews = None
        self.unconcatenated_rews = []
        self.concatenated_logp_as = None
        self.terminals = None

    def get_data(self):
        return self.obs, self.acts, self.concatenated_rews, self.unconcatenated_rews, self.concatenated_logp_as, \
               self.terminals
