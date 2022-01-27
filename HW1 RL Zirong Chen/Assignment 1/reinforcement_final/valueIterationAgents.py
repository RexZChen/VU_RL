# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


def _getQVal(mdp: mdp.MarkovDecisionProcess, values, discount, state, action):
    res = 0

    for next_state, prob in mdp.getTransitionStatesAndProbs(state, action):
        res += prob * (mdp.getReward(state, action, next_state) + discount * values[next_state])

    return res


def _getQVals(mdp: mdp.MarkovDecisionProcess, values, discount, state):
    res = []

    for action in mdp.getPossibleActions(state):
        res.append((_getQVal(mdp, values, discount, state, action), action))

    return res
# references:
# https://inst.eecs.berkeley.edu/~cs188/fa20/project3/
# https://github.com/janluke/cs188
# http://rail.eecs.berkeley.edu/deeprlcourse/


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        # self.values = util.Counter()  # A Counter is a dict with default 0
        self.values = collections.defaultdict(float)
        self.policy = None
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        discount = self.discount

        states = mdp.getStates()

        prevValues = collections.defaultdict(int)
        currValues = self.values

        for k in range(self.iterations):
            prevValues, currValues = currValues, prevValues

            for state in states:
                qValues = _getQVals(mdp, prevValues, discount, state)

                if not qValues:
                    currValues[state] = 0
                else:
                    maxQValue = max(qValues)[0]
                    currValues[state] = maxQValue

        self.values = currValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return _getQVal(self.mdp, self.values, self.discount, state, action)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        if self.policy is None:
            self.policy = policy = dict()

            for state in self.mdp.getStates():
                qValues = _getQVals(self.mdp, self.values, self.discount, state)

                if qValues:
                    policy[state] = max(qValues)[1]
                else:
                    policy[state] = None

        return self.policy[state]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
