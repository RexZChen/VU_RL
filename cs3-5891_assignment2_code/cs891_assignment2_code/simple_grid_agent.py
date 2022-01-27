import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import collections


def prob_s_prime_r_given_s_a(episode):
    return 0.8


def backward_computation(episode, gamma):
    Q = collections.defaultdict(lambda: collections.defaultdict(float))
    V = collections.defaultdict(float)

    # episode is a whole trajectory generated from env with "done" statement at the end
    # each item: (state, action, reward, next_state, done)
    episode = episode[::-1]

    # assert episode[0][4] == True
    # The last state should be assigned "True" as done

    terminal_state = episode[0][3]

    # by definition, the (Q) value of the terminal state should be 0
    # and there is no further action given the terminal state
    V[terminal_state] = 0.0

    for state, action, reward, next_state, done in episode:
        Q[terminal_state][action] = 0.0

    for state, action, reward, next_state, done in episode:
        Q[state][action] += prob_s_prime_r_given_s_a(episode) * (reward + gamma * V[next_state])

    for state, action, reward, next_state, done in episode:
        V[state] += Q[state][action]

    return Q, V


class GridworldAgent:
    def __init__(self, env, policy, gamma=0.9,
                 start_epsilon=0.9, end_epsilon=0.1, epsilon_decay=0.9):
        self.env = env
        self.n_action = len(self.env.action_space)
        self.policy = policy
        self.gamma = gamma
        self.v = dict.fromkeys(self.env.state_space, 0)  # state value initiated as 0
        self.n_v = dict.fromkeys(self.env.state_space,
                                 0)  # number of actions performed: use it for MC state value prediction
        self.q = defaultdict(lambda: np.zeros(self.n_action))  # action value
        self.n_q = defaultdict(
            lambda: np.zeros(self.n_action))  # number of actions performed: use it for MC state-action value prediction

        # epsilon greedy parameters
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay

    def get_epsilon(self, n_episode):
        epsilon = max(self.start_epsilon * (self.epsilon_decay ** n_episode), self.end_epsilon)
        return epsilon

    def get_v(self, start_state, epsilon=0.):
        episode = self.run_episode(start_state, epsilon)
        """
        Write the code to calculate and return the state value 
        given a deterministic policy. Should return a scalar. Study the components of episode to
        understand how to calculate return.
        YOUR CODE HERE
        """
        G = 0
        cnt = 0

        for state, _, reward, _, _ in episode[::-1]:
            G = self.gamma * G + reward

            if state == start_state:
                cnt += 1

        return G / cnt

        ######################################################################
        # Or I overthought this question, all I need to do is to return self.v
        # return self.v
        # raise NotImplementedError

    def get_q(self, start_state, first_action, epsilon=0.):
        episode = self.run_episode(start_state, epsilon, first_action)
        """
        Write the code to calculate and return the action value of a state 
        given a deterministic policy. Should return a scalar. Study the components of episode to
        understand how to calculate return.
        YOUR CODE HERE
        """
        G = 0
        cnt = 0

        for state, action, reward, _, _ in episode[::-1]:
            G = self.gamma * G + reward

            if (state, action) == (start_state, first_action):
                cnt += 1

        return G / cnt

        ######################################################################
        # Or I overthought this question, all I need to do is to return self.q
        # return self.q
        # raise NotImplementedError

    def select_action(self, state, epsilon):
        best_action = self.policy[state]
        if random.random() > epsilon:
            action = best_action
        else:
            action = np.random.choice(np.arange(self.n_action))
        return action

    def print_policy(self):
        for i in range(self.env.sz[0]):
            print('\n----------')
            for j in range(self.env.sz[1]):
                p = self.policy[(i, j)]
                out = self.env.action_text[p]
                print(f'{out} |', end='')

    def print_v(self, decimal=1):
        for i in range(self.env.sz[0]):
            print('\n---------------')
            for j in range(self.env.sz[1]):
                out = np.round(self.v[(i, j)], decimal)
                print(f'{out} |', end='')

    def run_episode(self, start, epsilon, first_action=None):
        result = []
        state = self.env.reset(start)
        # dictate first action to iterate q
        if first_action is not None:
            action = first_action
            next_state, reward, done, _ = self.env.step(action)
            result.append((state, action, reward, next_state, done))
            state = next_state
            if done: return result
        while True:
            action = self.select_action(state, epsilon)
            next_state, reward, done, _ = self.env.step(action)
            result.append((state, action, reward, next_state, done))
            state = next_state
            if done: break
        return result

    def update_policy_q(self):
        for state in self.env.state_space:
            self.policy[state] = np.argmax(self.q[state])

    def mc_predict_v(self, n_episode=10000, first_visit=True):
        for t in range(n_episode):

            traversed = []
            e = self.get_epsilon(t)
            # Generate an episode following pi: S0,A0,R1,S1,A1,R2...S(T-1),A(T-1),R(T)
            transitions = self.run_episode(self.env.start, e)
            states, actions, rewards, next_states, dones = zip(*transitions)

            states, actions, rewards, next_states = states[::-1], actions[::-1], rewards[::-1], next_states[::-1]

            # returns = collections.defaultdict(list)
            G = 0

            for i in range(len(transitions)):
                # Loop for each step of episode
                if first_visit and (states[i] not in traversed):
                    """
                    Implement first-visit Monte Carlo for state values(see Sutton and Barto Section 5.1)
                    Comment each line of code with what part of the pseudocode you are implementing in that line
                    YOUR CODE HERE
                    """
                    G = self.gamma * G + rewards[i]

                    # Append G to returns(S_t)
                    self.v[states[i]] += G

                    # assert len(returns[states[i]]) == 1

                    # Update V(S_t) using average(returns(S_t))
                    # self.v[states[i]] = sum(returns[states[i]]) / len(returns[states[i]])
                    self.n_v[states[i]] += 1

                    # By appending states[i] to traversed, it will make sure every state will only be visited once
                    traversed.append(states[i])

                elif not first_visit:
                    """
                    Implement any-visit Monte Carlo for state values(see Sutton and Barto Section 5.1)
                    Comment each line of code with what part of the pseudocode you are implementing in that line
                    YOUR CODE HERE
                    """
                    G = self.gamma * G + rewards[i]

                    # Append G to returns(S_t)
                    self.v[states[i]] += G

                    # Update V(S_t) using average(returns(S_t))
                    # self.v[states[i]] = sum(returns[states[i]]) / len(returns[states[i]])

                    self.n_v[states[i]] += 1

                    # assert len(returns[states[i]]) >= 1

        for state in self.env.state_space:
            if state != self.env.goal:
                self.v[state] = self.v[state] / self.n_v[state]
            else:
                self.v[state] = 0

    def mc_predict_q(self, n_episode=10000, first_visit=True):
        for t in range(n_episode):
            traversed = []
            e = self.get_epsilon(t)
            transitions = self.run_episode(self.env.start, e)
            states, actions, rewards, next_states, dones = zip(*transitions)

            states, actions, rewards, next_states = states[::-1], actions[::-1], rewards[::-1], next_states[::-1]

            # returns = collections.defaultdict(list)

            G = 0

            for i in range(len(transitions)):

                if first_visit and ((states[i], actions[i]) not in traversed):
                    """
                    Implement first-visit Monte Carlo for state-action values(see Sutton and Barto Section 5.2)
                    Comment each line of code with what part of the pseudocode you are implementing in that line
                    YOUR CODE HERE
                    """
                    G = self.gamma * G + rewards[i]

                    # append G to returns(S_t, A_t)
                    # assert len(returns[(states[i], actions[i])]) == 1

                    # Update Q(S_t, A_t) using average(returns(S_t, A_t))
                    # self.q[states[i]][actions[i]] = sum(returns[(states[i], actions[i])]) / len(
                    #     returns[(states[i], actions[i])])
                    self.q[states[i]][actions[i]] += G
                    self.n_q[states[i]][actions[i]] += 1

                    # By appending (state,action) pairs to traversed,
                    # it will make sure every pair will only be visited once
                    traversed.append((states[i], actions[i]))

                elif not first_visit:

                    """
                    Implement any-visit Monte Carlo for state-action values(see Sutton and Barto Section 5.2)
                    Comment each line of code with what part of the pseudocode you are implementing in that line
                    YOUR CODE HERE
                    """
                    G = self.gamma * G + rewards[i]

                    # append G to returns(S_t, A_t)

                    # Update Q(S_t, A_t) using average(returns(S_t, A_t))
                    # self.q[states[i]][actions[i]] = sum(returns[(states[i], actions[i])]) / len(
                    #     returns[(states[i], actions[i])])
                    self.q[states[i]][actions[i]] += G
                    self.n_q[states[i]][actions[i]] += 1

                    # assert len(returns[(states[i], actions[i])]) >= 1

        for state in self.env.state_space:
            for action in range(self.n_action):
                if state != self.env.goal:
                    self.q[state][action] = self.q[state][action] / self.n_q[state][action]
                else:
                    self.q[state][action] = 0

    def mc_predict_q_glie(self, n_episode=10000, first_visit=True, lr=0.0):
        for t in range(n_episode):
            traversed = []
            e = self.get_epsilon(t)
            transitions = self.run_episode(self.env.start, e)

            states, actions, rewards, next_states, dones = zip(*transitions)
            states, actions, rewards, next_states = states[::-1], actions[::-1], rewards[::-1], next_states[::-1]

            G = 0

            for i in range(len(transitions)):

                if first_visit and ((states[i], actions[i]) not in traversed):
                    G = self.gamma * G + rewards[i]
                    self.n_q[states[i]][actions[i]] += 1

                    learning_factor = float(1/self.n_q[states[i]][actions[i]]) if lr == 0 else lr

                    self.q[states[i]][actions[i]] += (G - self.q[states[i]][actions[i]]) * learning_factor

                    traversed.append((states[i], actions[i]))

                elif not first_visit:
                    G = self.gamma * G + rewards[i]
                    self.n_q[states[i]][actions[i]] += 1

                    learning_factor = float(1/self.n_q[states[i]][actions[i]]) if lr == 0 else lr

                    self.q[states[i]][actions[i]] += (G - self.q[states[i]][actions[i]]) * learning_factor

        for state in self.env.state_space:
            for action in range(self.n_action):
                if state != self.env.goal:
                    self.q[state][action] = self.q[state][action] / self.n_q[state][action]
                else:
                    self.q[state][action] = 0

    def mc_control_q(self, n_episode=10000, first_visit=True):
        """
        Write the code to perform Monte Carlo Control for state-action values
        Hint: You just need to do prediction then update the policy
        YOUR CODE HERE
        """
        # generate trajectory and predictions
        self.mc_predict_q(n_episode=n_episode, first_visit=first_visit)
        # update policy
        self.update_policy_q()
        # raise NotImplementedError

    def mc_control_glie(self, n_episode=10000, first_visit=True, lr=0.6):
        """
        Bonus: Taking hints from the mc_predict_q and mc_control_q methods, write the code to perform GLIE Monte
        Carlo control. Comment each line of code with what part of the pseudocode you are implementing in that line
        YOUR CODE HERE
        """
        # generate trajectory
        self.mc_predict_q_glie(n_episode=n_episode, first_visit=first_visit, lr=lr)
        # update policy
        self.update_policy_q()
        # raise NotImplementedError
