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
from bandit_problem.algorithm.random_bandit_algorithm import RandomBanditAlgorithm

import numpy as np


class EpsilonGreedyBanditAlgorithm(GreedyBanditAlgorithm,
                                   RandomBanditAlgorithm):
    """
    Epsilon-greedy Bandit Algorithm

    Parameters
    ----------
    n_arms : int
        Number of arms
    epsilon : float
        Probability to choose an action at random
    """
    def __init__(self, n_arms=10, epsilon=0.1):
        GreedyBanditAlgorithm.__init__(self, n_arms=n_arms)
        self.epsilon = epsilon

    def get_action(self):
        """
        Get Epsilon-greedy action

        Choose an action at random with probability epsilon and a greedy
        action otherwise.

        Returns
        -------
        int
            The chosen action
        """
        if np.random.rand() < self.epsilon:
            return RandomBanditAlgorithm.get_action(self)
        else:
            return GreedyBanditAlgorithm.get_action(self)
