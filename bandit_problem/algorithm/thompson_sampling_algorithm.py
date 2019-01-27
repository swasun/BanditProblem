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


class ThompsonSamplingAlgorithm(BanditAlgorithm):

    def __init__(self, n_arms):
        BanditAlgorithm.__init__(self, n_arms=n_arms)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def get_action(self):
        theta = np.random.beta(self.alpha, self.beta)
        return np.argmax(theta)

    def fit_step(self, action, reward):
        self.alpha[action] += reward
        self.beta[action] += 1 - reward
