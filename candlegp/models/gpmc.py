# Copyright 2016 James Hensman, alexggmatthews
# Copyright 2017 Thomas Viehmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torch.autograd import Variable

from .. import likelihoods
from .. import densities
from .. import parameter
from .. import priors
from .. import conditionals

from .model import GPModel


class GPMC(GPModel):
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=None,
                 num_latent=None,
                 **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood, mean_function are appropriate GPflow objects

        This is a vanilla implementation of a GP with a non-Gaussian
        likelihood. The latent function values are represented by centered
        (whitened) variables, so

            v ~ N(0, I)
            f = Lv + m(x)

        with

            L L^T = K

        """
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.num_data = X.size(0)
        self.num_latent = num_latent or Y.size(1)
        self.V = parameter.Param(self.X.data.new(self.num_data, self.num_latent).zero_())
        self.V.prior = priors.Gaussian(0., 1.)

    def compute_log_likelihood(self):
        """
        Construct a tf function to compute the likelihood of a general GP
        model.

            \log p(Y, V | theta).

        """
        K = self.kern.K(self.X)
        L = torch.potrf(
            K + Variable(torch.eye(self.X.size(0), out=K.data.new()) * self.jitter_level), upper=False)
        F = torch.matmul(L, self.V.get()) + self.mean_function(self.X)

        return self.likelihood.logp(F, self.Y).sum()

    def predict_f(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (F=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at X.

        """
        mu, var = conditionals.conditional(Xnew, self.X, self.kern, self.V,
                              full_cov=full_cov,
                              q_sqrt=None, whiten=True)
        return mu + self.mean_function(Xnew), var
