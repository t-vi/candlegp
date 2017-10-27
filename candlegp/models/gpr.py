# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, fujiisoup
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

from .model import GPModel

class GPR(GPModel):
    """
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood i this models is sometimes referred to as the 'marginal log likelihood', and is given by

    .. math::

       \\log p(\\mathbf y \\,|\\, \\mathbf f) = \\mathcal N\\left(\\mathbf y\,|\, 0, \\mathbf K + \\sigma_n \\mathbf I\\right)
    """
    def __init__(self, X, Y, kern, mean_function=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian(ttype=type(X.data))
        super(GPR,self).__init__(X, Y, kern, likelihood, mean_function, **kwargs)
        self.num_latent = Y.size(1)

    def compute_log_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        K = self.kern.K(self.X) + Variable(torch.eye(self.X.size(0),out=self.X.data.new())) * self.likelihood.variance.get()
        L = torch.potrf(K, upper=False)
        m = self.mean_function(self.X)
        return densities.multivariate_normal(self.Y, m, L)

    def predict_f(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X) + Variable(torch.eye(self.X.size(0),out=self.X.data.new())) * self.likelihood.variance.get()
        L = torch.potrf(K, upper=False)
        A,_ = torch.gesv(Kx, L)  # could use triangular solve, note gesv has B first, then A in AX=B
        V,_ = torch.gesv(self.Y - self.mean_function(self.X),L) # could use triangular solve
        fmean = torch.mm(A.t(), V) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - torch.mm(A.t(), A)
            fvar = fvar.unsqueeze(2).expand(fvar.size(0), fvar.size(1), self.Y.size(1))
        else:
            fvar = self.kern.Kdiag(Xnew) - (A**2).sum(0)
            fvar = fvar.view(-1,1)
            fvar = fvar.expand(fvar.size(0),self.Y.size(1))
        return fmean, fvar
