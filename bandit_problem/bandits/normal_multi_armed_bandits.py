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

import numpy as np


class NormalMultiArmedBandits:
    """
    Bandit problem with normal distributions

    Parameters
    ----------
    n_arms : int
        Number of arms or actions
    """

    def __init__(self, n_arms):
        self._true_values = np.random.randn(n_arms)

    @property
    def n_arms(self):
        return self._true_values.size

    def step(self, a):
        assert 0 <= a
        assert a < self.n_arms
        return np.random.randn() + self._true_values[a]

    def __str__(self):
        return '{}-arms bandit with Normal distributions'.format(self.n_arms)
