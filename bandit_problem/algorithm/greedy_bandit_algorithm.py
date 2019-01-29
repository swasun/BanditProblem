 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2018 Charly Lamothe                                                 #
 #                                                                                   #
 # This file is part of BanditProblem.                                               #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

from bandit_problem.algorithm.bandit_algorithm import BanditAlgorithm

import numpy as np


class GreedyBanditAlgorithm(BanditAlgorithm):
    """
    Greedy Bandit Algorithm

    Parameters
    ----------
    n_arms : int
        Number of arms
    """
    def __init__(self, n_arms=10):
        BanditAlgorithm.__init__(self, n_arms=n_arms) # Estimation of the value of each arm
        self._value_estimates = np.zeros(n_arms)
        """
        Number of times each arm has been chosen
        in order to estimate the mean
        """
        self._n_estimates = np.zeros(n_arms)

    def get_action(self):
        """
        Choose the action with maximum estimated value

        Returns
        -------
        int
            The chosen action
        """
        return np.argmax(self._value_estimates)

    def fit_step(self, action, reward):
        """
        Update current value estimates with an (action, reward) pair

        $$
        \mu_{n+1} = \mu_n + \frac{1}{n+1}(x_{n+1} - \mu_n)
        $$
        with $x_{n+1}$ the reward

        Greedy will be bad because it exploit too much
        and doesn't explore enough (in a way, we're stucked).
        This is way we add an epsilon factor in epsilon-greedy
        version. So the greedy isn't used in practice.

        Parameters
        ----------
        action : int
        reward : float

        """
        self._n_estimates[action] += 1
        q = self._value_estimates[action]
        n = self._n_estimates[action]
        self._value_estimates[action] += (reward - q) / n
