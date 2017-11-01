# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, Mark van der Wilk
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


import numpy
import torch
from torch.autograd import Variable

from .. import conditionals
from .. import kullback_leiblers
from .. import parameter

from .model import GPModel


class SVGP(GPModel):
    """
    This is the Sparse Variational GP (SVGP). The key reference is

    ::

      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews,
                Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }

    """
    def __init__(self, X, Y, kern, likelihood, Z,
                 mean_function=None,
                 num_latent=None,
                 q_diag=False,
                 whiten=True,
                 minibatch_size=None,
                 **kwargs):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        - kern, likelihood, mean_function are appropriate GPflow objects
        - Z is a matrix of pseudo inputs, size M x D
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        """
        # sort out the X, Y into MiniBatch objects if required.
        self.minibatch_size = minibatch_size

        # init the super class, accept args
        super(SVGP, self).__init__(X, Y, kern, likelihood, mean_function, **kwargs)
        self.num_data = X.size(0)
        self.q_diag, self.whiten = q_diag, whiten
        self.Z = parameter.Param(Z)
        self.num_latent = num_latent or Y.size(1)
        self.num_inducing = Z.size(0)

        # init variational parameters
        self.q_mu = parameter.Param(self.Z.data.new(self.num_inducing, self.num_latent).zero_())
        if self.q_diag:
            self.q_sqrt = parameter.PositiveParam(self.Z.data.new(self.num_inducing, self.num_latent).fill_(1.0))
        else:
            q_sqrt = torch.eye(self.num_inducing, out=self.Z.data.new()).unsqueeze(2).expand(-1,-1,self.num_latent)
            self.q_sqrt = parameter.LowerTriangularParam(q_sqrt) # should the diagonal be all positive?

    def prior_KL(self):
        if self.whiten:
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_white_diag(self.q_mu.get(), self.q_sqrt.get())
            else:
                KL = kullback_leiblers.gauss_kl_white(self.q_mu.get(), self.q_sqrt.get())
        else:
            K = self.kern.K(self.Z.get()) + torch.eye(self.num_inducing, out=self.Z.new()) * self.jitter_level
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_diag(self.q_mu.get(), self.q_sqrt.get(), K)
            else:
                KL = kullback_leiblers.gauss_kl(self.q_mu.get(), self.q_sqrt.get(), K)
        return KL

    def compute_log_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.prior_KL()

        # Get conditionals
        fmean, fvar = self.predict_f(self.X, full_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = float(self.num_data) / self.X.size(0)

        return var_exp.sum() * scale - KL

    def predict_f(self, Xnew, full_cov=False):
        mu, var = conditionals.conditional(Xnew, self.Z.get(), self.kern, self.q_mu,
                                           q_sqrt=self.q_sqrt.get(), full_cov=full_cov, whiten=self.whiten,
                                           jitter_level=self.jitter_level)
        return mu + self.mean_function(Xnew), var
