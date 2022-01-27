import numpy as np
from collections import defaultdict
import random


class TaxiAgent:
    def __init__(self, env, gamma=0.8, alpha=1e-1,
                 start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999):

        self.env = env
        self.n_action = self.env.action_space.n
        self.gamma = gamma
        self.alpha = alpha

        # action values
        self.q = defaultdict(lambda: np.zeros(self.n_action))  # action value

        # epsilon greedy parameters
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay

    # get epsilon
    def get_epsilon(self, n_episode):
        epsilon = max(self.start_epsilon * (self.epsilon_decay ** n_episode), self.end_epsilon)
        return epsilon

    # select action based on epsilon greedy
    def select_action(self, state, epsilon):
        # implicit policy; if we have action values for that state, choose the largest one, else random
        best_action = np.argmax(self.q[state]) if state in self.q else self.env.action_space.sample()
        if random.random() > epsilon:
            action = best_action
        else:
            action = self.env.action_space.sample()
        return action

    def on_policy_td_sarsa(self, state, action, reward, next_state, n_episode):
        """
        Implement On policy TD learning or SARSA
        YOUR CODE HERE
        """
        # everytime when this function gets called,
        # 1. it's already in the training loop
        # 2. action A, state S, next_state S' are given, no need to initialize S nor choose A
        # 3. n_episode is given
        # 4. update Q value from A and A'
        # 5. no need to update S' nor A'

        # loop for each step of episode
        # take action A, observe R, S'
        A = action
        S = state
        R = reward
        S_prime = next_state
        # choose A' from S' using policy derived from Q
        A_prime = self.select_action(state=S_prime, epsilon=self.get_epsilon(n_episode))
        # update Q
        self.q[S][A] = self.q[S][A] + self.alpha * (
            R + self.gamma * self.q[S_prime][A_prime] - self.q[S][A]
        )
        # update S <- S', A <- A'
        # S = S_prime
        # A = A_prime
        # raise NotImplementedError

    def off_policy_td_q_learning(self, state, action, reward, next_state):
        """
        Implement Off policy TD learning ie SARSA-MAX/Q learning 
        YOUR CODE HERE
        """
        # Step 1: Give state s_t, choose an action a_t using epsilon greedy policy
        # Step 2: Update the action value Q(s_t, a_t) using the equation

        # everytime when this function gets called,
        # 1. it's already in the training loop
        # 2. action A, state S, next_state S' are given, no need to initialize S nor choose A
        # 3. n_episode is given
        # 4. update Q value from A and A'
        # 5. no need to update S' nor A'

        A = action
        S = state
        R = reward
        S_prime = next_state

        # update Q(S,A)
        self.q[S][A] = self.q[S][A] + self.alpha * (
            R + self.gamma * max(self.q[S_prime]) - self.q[S][A]
        )
        # raise NotImplementedError
