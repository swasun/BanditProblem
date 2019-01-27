 ###############################################################################
 # Copyright (C) 2019 Charly Lamothe                                           #
 #                                                                             #
 # This file is part of BanditProblem.                                         #
 #                                                                             #
 #   Licensed under the Apache License, Version 2.0 (the "License");           #
 #   you may not use this file except in compliance with the License.          #
 #   You may obtain a copy of the License at                                   #
 #                                                                             #
 #   http://www.apache.org/licenses/LICENSE-2.0                                #
 #                                                                             #
 #   Unless required by applicable law or agreed to in writing, software       #
 #   distributed under the License is distributed on an "AS IS" BASIS,         #
 #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
 #   See the License for the specific language governing permissions and       #
 #   limitations under the License.                                            #
 ###############################################################################

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
