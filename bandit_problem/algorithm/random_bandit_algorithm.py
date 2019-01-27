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


class RandomBanditAlgorithm(BanditAlgorithm):
    """
    A generic class for Bandit Algorithms

    Parameters
    ----------
    n_arms : int
        Number of arms
    """
    def __init__(self, n_arms=10):
        BanditAlgorithm.__init__(self, n_arms=n_arms)
        self._value_estimates = np.zeros(n_arms) # Estimation of the value of each arm
        self._n_estimates = np.zeros(n_arms) # Number of times each arm has been chosen
        self.n_arms = n_arms

    def get_action(self):
        """
        Choose an action at random uniformly among the available arms

        Returns
        -------
        int
            The chosen action
        """
        return np.random.randint(low=0, high=self.n_arms, dtype=int)

    def fit_step(self, action, reward):
        """
        Do nothing since actions are chosen at random

        Parameters
        ----------
        action : int
        reward : float

        """
        pass
