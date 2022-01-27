"""
Abstract class for the Agent
All agents will inherit from this class

Author: Zirong Chen
"""

class Agent:

    def __init__(self):
        pass

    def train(self, *args):
        raise NotImplementedError