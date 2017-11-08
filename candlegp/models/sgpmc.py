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


class SGPMC(GPModel):
    """
    This is the Sparse Variational GP using MCMC (SGPMC). The key reference is

    ::

      @inproceedings{hensman2015mcmc,
        title={MCMC for Variatinoally Sparse Gaussian Processes},
        author={Hensman, James and Matthews, Alexander G. de G.
                and Filippone, Maurizio and Ghahramani, Zoubin},
        booktitle={Proceedings of NIPS},
        year={2015}
      }

    The latent function values are represented by centered
    (whitened) variables, so

    .. math::
       :nowrap:

       \\begin{align}
       \\mathbf v & \\sim N(0, \\mathbf I) \\\\
       \\mathbf u &= \\mathbf L\\mathbf v
       \\end{align}

    with

    .. math::
        \\mathbf L \\mathbf L^\\top = \\mathbf K


    """
    def __init__(self, X, Y, kern, likelihood, Z,
                 mean_function=None,
                 num_latent=None,
                 **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a data matrix, of inducing inputs, size M x D
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
        self.num_inducing = Z.size(0)
        self.Z = parameter.Param(Z)
        self.V = parameter.Param(self.X.data.new(self.num_inducing, self.num_latent).zero_())
        self.V.prior = priors.Gaussian(self.X.data.new(1).fill_(0.), self.X.data.new(1).fill_(1.))

    def compute_log_likelihood(self):
        """
        Construct a tf function to compute the likelihood of a general GP
        model.

            \log p(Y, V | theta).

        """
        # get the (marginals of) q(f): exactly predicting!
        fmean, fvar = self.predict_f(self.X, full_cov=False)
        return self.likelihood.variational_expectations(fmean, fvar, self.Y).sum()

    def predict_f(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (F=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at X.

        """
        mu, var = conditionals.conditional(Xnew, self.Z.get(), self.kern, self.V.get(),
                              full_cov=full_cov, q_sqrt=None, whiten=True)
        return mu + self.mean_function(Xnew), var
