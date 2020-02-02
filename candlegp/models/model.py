# Copyright 2016 James Hensman, Mark van der Wilk, Valentine Svensson, alexggmatthews, fujiisoup
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

import abc
import torch
import numpy

from .. import mean_functions
from .. import parameter

class GPModel(torch.nn.Module):
    """
    A base class for Gaussian process models, that is, those of the form

    .. math::
       :nowrap:

       \\begin{align}
       \\theta & \sim p(\\theta) \\\\
       f       & \sim \\mathcal{GP}(m(x), k(x, x'; \\theta)) \\\\
       f_i       & = f(x_i) \\\\
       y_i\,|\,f_i     & \sim p(y_i|f_i)
       \\end{align}

    This class mostly adds functionality to compile predictions. To use it,
    inheriting classes must define a build_predict function, which computes
    the means and variances of the latent function. This gets compiled
    similarly to build_likelihood in the Model class.

    These predictions are then pushed through the likelihood to obtain means
    and variances of held out data, self.predict_y.

    The predictions can also be used to compute the (log) density of held-out
    data via self.predict_density.

    For handling another data (Xnew, Ynew), set the new value to self.X and self.Y

    >>> m.X = Xnew
    >>> m.Y = Ynew
    """

    def __init__(self, X, Y, kern, likelihood, mean_function, name=None, jitter_level=1e-6):
        super(GPModel, self).__init__()
        self.name = name
        self.mean_function = mean_function or mean_functions.Zero()
        self.kern = kern
        self.likelihood = likelihood
        self.jitter_level = jitter_level

        if isinstance(X, numpy.ndarray):
            # X is a data matrix; each row represents one instance
            X = torch.from_numpy(X)
        if isinstance(Y, numpy.ndarray):
            # Y is a data matrix, rows correspond to the rows in X,
            # columns are treated independently
            Y = torch.from_numpy(Y)
        self.X, self.Y = X, Y

    @abc.abstractmethod
    def compute_log_prior(self):
        """Compute the log prior of the model."""
        pass

    @abc.abstractmethod
    def compute_log_likelihood(self, X=None, Y=None):
        """Compute the log likelihood of the model."""
        pass

    def objective(self, X=None, Y=None):
        pos_objective = self.compute_log_likelihood(X, Y)
        for param in self.parameters():
            if isinstance(param, parameter.ParamWithPrior):
                pos_objective = pos_objective + param.get_prior()
        return -pos_objective

    def forward(self, X=None, Y=None):
        return self.objective(X, Y)
    
    @abc.abstractmethod
    def predict_f(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        pass
        #return self._build_predict(Xnew) #    @autoflow((settings.tf_float, [None, None]))

    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        return self.predict_f(Xnew, full_cov=True)

    def predict_f_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.predict_f(Xnew, full_cov=True)
        jitter = torch.eye(mu.size(0), dtype=mu.dtype, device=mu.device) * self.jitter_level
        samples = []
        for i in range(self.num_latent): # TV-Todo: batch??
            L = torch.cholesky(var[:, :, i] + jitter, upper=False)
            V = torch.randn(L.size(0), num_samples, dtype=L.dtype, device=L.device)
            samples.append(mu[:, i:i + 1] + torch.matmul(L, V))
        return torch.stack(samples, dim=0) # TV-Todo: transpose?

    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self.predict_f(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self.predict_f(Xnew)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)

    def _repr_html_(self):
        s =  'Model {}<ul>'.format(type(self).__name__)
        for n,c in self.named_children():
            s += '<li>{}: {}</li>'.format(n, type(c).__name__)
        s += '</ul><table><tr><th>Parameter</th><th>Value</th><th>Prior</th><th>ParamType</th></tr><tr><td>'
        s += '</td></tr><tr><td>'.join(['</td><td>'.join((n,str(p.get().data.cpu().numpy()),str(p.prior),type(p).__name__)) for n,p in self.named_parameters()])
        s += '</td></tr></table>'
        return s
