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

from bandit_problem.algorithm.greedy_bandit_algorithm import GreedyBanditAlgorithm

import numpy as np


class UcbBanditAlgorithm(GreedyBanditAlgorithm):
    """
    $$
    P(\mu \lt \frac{1}{n} \sum^n_{i=1} X_i + \varepsilon) \gt 1-\delta
    $$
    with $\frac{1}{n} \sum^n_{i=1} X_i + \varepsilon$ is $u$
    and $1-\delta$ is equal to 95%. More and more we increase
    in time and at most we are sure of our result.

    Parameters
    ----------
    n_arms : int
        Number of arms
    c : float
        Positive parameter to adjust exploration/explotation UCB criterion
    """
    def __init__(self, n_arms, c):
        GreedyBanditAlgorithm.__init__(self, n_arms=n_arms)
        self.n_arms = n_arms
        self.c = c

    def get_action(self):
        """
        Get UCB action

        Returns
        -------
        int
            The chosen action
        """
        return np.argmax(self.get_upper_confidence_bound())
        
    def get_upper_confidence_bound(self):
        return np.argmax(
            [self._value_estimates[action]
             + self.c * np.sqrt(np.log(np.sum(self._n_estimates))/self._n_estimates[action])
             for action in range(self.n_arms)])
