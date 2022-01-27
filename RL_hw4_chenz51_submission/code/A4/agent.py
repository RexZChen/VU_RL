"""
Abstract class for agents in future inheritance

Author: Zirong Chen
"""


class Agent:

    def __init__(self):
        pass

    def choose_action(self, *args):
        raise NotImplementedError

    def train(self, *args):
        raise NotImplementedError
