
# Include your imports here, if any are used.
import math
import copy
student_name = "Jonathan Kim"



class ValueIterationAgent:
    """Implement Value Iteration Agent using Bellman Equations."""

    def __init__(self, game, discount):
        """Store game object and discount value into the agent object,
        initialize values if needed.
        """
        # game attribute
        self.game = game
        # discount factor attribute
        self.discount = discount
        # dictionary attribute: we know that states are hashable
        self.values = {}
        # initialize values as 0, we haven't iterated yet
        for state in self.game.states:
            self.values[state] = 0

    def get_value(self, state):
        """Return value V*(s) correspond to state.
        State values should be stored directly for quick retrieval.
        """
        # if the state is in the dictionary
        if state in self.values:
            # return the value of that state
            return self.values[state]
        # else it is a terminal state
        else:
            # otherwise we know it's terminal, return 0 as its value
            return 0

    def get_q_value(self, state, action):
        """Return Q*(s,a) correspond to state and action.
        Q-state values should be computed using Bellman equation:
        Q*(s,a) = Σ_s' T(s,a,s') [R(s,a,s') + γ V*(s')]
        """
        # set q value as 0 for initialization
        q = 0
        # get mappings of state to actions
        stateMap = self.game.get_transitions(state, action)
        # iterate through this mapping
        for next, v in stateMap.items():
            # add the prob weighted (v) value to q value for summing
            q += v * (self.game.get_reward(state, action, next)
                      + (self.get_value(next) * self.discount))
        # return q value after iteration
        return q  # TODO

    def get_best_policy(self, state):
        """Return policy π*(s) correspond to state.
        Policy should be extracted from Q-state values using policy extraction:
        π*(s) = argmax_a Q*(s,a)
        """
        # best action is none
        bestAction = None
        # set maximum q value to 0 initially
        maxQ = -math.inf
        # iterate through all potential actions
        for action in self.game.get_actions(state):
            # get the q value
            q = self.get_q_value(state, action)
            # if this is the maximum q value
            if maxQ < q:
                # set the action to best action
                maxQ = q
                bestAction = action
        # return the best action
        return bestAction  # TODO

    def iterate(self):
        """Run single value iteration using Bellman equation:
        V_{k+1}(s) = max_a Q*(s,a)
        Then update values: V*(s) = V_{k+1}(s)
        """
        # iterate through the q values, qval is storage
        new_values = copy.deepcopy(self.values)
        # iterate through states
        for state in self.game.states:
            # get the best policy
            action = self.get_best_policy(state)
            # use policy to get the q value max
            q = self.get_q_value(state, action)
            # store said q value
            new_values[state] = q
        # set value
        self.values = new_values



class PolicyIterationAgent(ValueIterationAgent):
    """Implement Policy Iteration Agent.

    The only difference between policy iteration and value iteration is at
    their iteration method. However, if you need to implement helper function
    or override ValueIterationAgent's methods, you can add them as well.
    """

    def iterate(self):
        """Run single policy iteration.
        Fix current policy, iterate state values V(s) until
        |V_{k+1}(s) - V_k(s)| < ε
        """
        # epsilon value given
        epsilon = 1e-6
        # set of actions we'll iterate through n while loop
        actions = {}
        # initialize V_k(s) arbitrarily: policy evaluation
        for state in self.game.states:
            # get the best policy from the state
            actions[state] = self.get_best_policy(state)

        # iterate through states
        # iterate through the q values, qval is storage
        unchanged = True
        while unchanged:
            maxDelta = -10000000
            # iterate through states
            newValues = copy.deepcopy(self.values)
            for state in self.game.states:
                # get action from state
                action = actions[state]
                # set new q value to newvalues
                newValues[state] = self.get_q_value(state, action)
                # calculate delta: the difference between the two states
                delta = abs(newValues[state] - self.get_value(state))
                # get max delta
                maxDelta = max(maxDelta, delta)
            # set new values to self.values attribute
            self.values = newValues
            # if max delta is less than epsilon, meaning we have small error
            if maxDelta < epsilon:
                # unchanged is false
                unchanged = False