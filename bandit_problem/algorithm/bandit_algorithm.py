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

import abc


class BanditAlgorithm(abc.ABC):
    """
    A generic abstract class for Bandit Algorithms

    Parameters
    ----------
    n_arms : int
        Number of arms
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_arms=10):
        self.n_arms = n_arms

    @abc.abstractmethod
    def get_action(self):
        """
        Choose an action (abstract)

        Returns
        -------
        int
            The chosen action
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_step(self, action, reward):
        """
        Update current value estimates with an (action, reward) pair (abstract)

        Parameters
        ----------
        action : int
        reward : float

        """
        raise NotImplementedError
