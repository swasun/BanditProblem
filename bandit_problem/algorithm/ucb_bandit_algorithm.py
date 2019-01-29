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
