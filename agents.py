import random



class QLearningAgent:
    """Implement Q Reinforcement Learning Agent using Q-table."""

    def __init__(self, game, discount, learning_rate, explore_prob):
        """Store any needed parameters into the agent object.
        Initialize Q-table.
        """
        # create attributes for all inputs: game
        self.game = game
        # create attributes for all inputs: discount
        self.discount = discount
        # create attributes for all inputs: learning rate
        self.learning_rate = learning_rate
        # create attributes for all inputs: explore probability
        self.explore_prob = explore_prob
        # q table to store values: dictionary, (state, action) hash to value
        self.qvalue = {}

    def get_q_value(self, state, action):
        """Retrieve Q-value from Q-table.
        For an never seen (s,a) pair, the Q-value is by default 0.
        """
        # if the state, action pair is in our qtable
        if (state, action) in self.qvalue:
            # output its qvalue
            return self.qvalue[(state, action)]
        # otherwise, if it's not in our table
        else:
            # put it in our table, defaulted to 0
            self.qvalue[(state, action)] = 0
            # then return the value 0
            return 0

    def get_value(self, state):
        """Compute state value from Q-values using Bellman Equation.
        V(s) = max_a Q(s,a)
        """
        # return the max q value from all possible actions for a state
        return (max((self.get_q_value(state, action)
                     for action in self.game.get_actions(state)), default=0))

    def get_best_policy(self, state):
        """Compute the best action to take in the state
        using Policy Extraction.
        π(s) = argmax_a Q(s,a)

        If there are ties, return a random one for better performance.
        Hint: use random.choice().
        """
        # initialize maxQ value to negative infinity
        maxQ = float('-inf')
        # initialize best action to None: we don't know best action yet
        bestAction = None
        # iterate through all of the actions possible
        for action in self.game.get_actions(state):
            # get the qvalue
            q = self.get_q_value(state, action)
            # if this is our max qvalue
            if maxQ < q:
                # set the max qvalue to the qvalue computed
                maxQ = q
                # set the best action to the current action we're iterating on
                bestAction = action
            # if this action is equivalent to the max value
            elif maxQ == q:
                # then randomly choose from either action: must be a seq
                options = list([bestAction, action])
                bestAction = random.choice(options)
        # return the best action that we found.
        return bestAction

    def update(self, state, action, next_state, reward):
        """Update Q-values using running average.
        Q(s,a) = (1 - α) Q(s,a) + α (R + γ V(s'))
        Where α is the learning rate, and γ is the discount.

        Note: You should not call this function in your code.
        """
        # in our table, update our qvalue
        self.qvalue[(state, action)] = ((1 - self.learning_rate)
                                        * self.get_q_value(state, action)
                                        + self.learning_rate
                                        * (reward + (self.discount *
                                                     self.get_value(next_state
                                                                    ))))
        # return None
        return None

    def get_action(self, state):
        """Compute the action to take for the agent, incorporating exploration.
        That is, with probability ε, act randomly.
        Otherwise, act according to the best policy.

        Hint: use random.random() < ε to check if exploration is needed.
        """
        if random.random() < self.explore_prob:
            return random.choice(list(self.game.get_actions(state)))
        else:
            return self.get_best_policy(state)


class ApproximateQAgent(QLearningAgent):
    """Implement Approximate Q Learning Agent using weights."""

    def __init__(self, *args, extractor):
        """Initialize parameters and store the feature extractor.
        Initialize weights table."""
        # initialize extractor function attribute
        self.extractor = extractor
        # initialize weights table
        self.weights = {}
        # call QLearningAgent constructor
        super().__init__(*args)

    def get_weight(self, feature):
        """Get weight of a feature.
        Never seen feature should have a weight of 0.
        """
        # if the feature is in self.weights
        if feature in self.weights:
            # return the existing value stored in dictionary
            return self.weights[feature]
        # otherwise it's not in the table
        else:
            # set the weight to 0 in the weights table
            self.weights[feature] = 0
            # return 0, its initialized value
            return 0

    def get_q_value(self, state, action):
        """Compute Q value based on the dot product of feature
        components and weights.
        Q(s,a) = w_1 * f_1(s,a) + w_2 * f_2(s,a) + ... + w_n * f_n(s,a)
        """
        # initialize q value to 0
        q = 0
        # iterate through all feature values
        for feature, val in self.extractor(state, action).items():
            # add the weighted value of each feature to q
            q += self.get_weight(feature) * val
        # return the weighted, computed q
        return q  # TODO

    def update(self, state, action, next_state, reward):
        """Update weights using least-squares approximation.
        Δ = R + γ V(s') - Q(s,a)
        Then update weights: w_i = w_i + α * Δ * f_i(s, a)
        """
        # compute delta
        delta = (reward
                 + (self.discount * self.get_value(next_state))
                 - self.get_q_value(state, action))
        # update weights
        for feature, value in self.extractor(state, action).items():
            # add product of learning rate/delta/value to weight of feature
            self.weights[feature] = (self.get_weight(feature)
                                     + (self.learning_rate * delta * value))
