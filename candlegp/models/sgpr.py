# Copyright 2016 James Hensman, alexggmatthews, Mark van der Wilk
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

from .. import likelihoods
from .. import densities
from .. import parameter

from .model import GPModel


class SGPRUpperMixin(object):
    """
    Upper bound for the GP regression marginal likelihood.
    It is implemented here as a Mixin class which works with SGPR and GPRFITC.
    Note that the same inducing points are used for calculating the upper bound,
    as are used for computing the likelihood approximation. This may not lead to
    the best upper bound. The upper bound can be tightened by optimising Z, just
    as just like the lower bound. This is especially important in FITC, as FITC
    is known to produce poor inducing point locations. An optimisable upper bound
    can be found in https://github.com/markvdw/gp_upper.

    The key reference is

    ::

      @misc{titsias_2014,
        title={Variational Inference for Gaussian and Determinantal Point Processes},
        url={http://www2.aueb.gr/users/mtitsias/papers/titsiasNipsVar14.pdf},
        publisher={Workshop on Advances in Variational Inference (NIPS 2014)},
        author={Titsias, Michalis K.},
        year={2014},
        month={Dec}
      }
    """

    def compute_upper_bound(self):
        num_inducing = self.Z.size(0)
        num_data = self.Y.size(0)

        Kdiag = self.kern.Kdiag(self.X)
        jitter = Variable(torch.eye(num_inducing, out=self.Z.data.new())) * self.jitter_level
        Kuu = self.kern.K(self.Z.get()) + jitter
        Kuf = self.kern.K(self.Z.get(), self.X)

        L = torch.potrf(Kuu, upper=False)
        LB = torch.potrf(Kuu + self.likelihood.variance.get() ** -1.0 * torch.matmul(Kuf, Kuf.t()), upper=False)

        LinvKuf, _ = torch.gesv(Kuf, L) # could use triangular solve
        # Using the Trace bound, from Titsias' presentation
        c = Kdiag.sum() - (LinvKuf ** 2.0).sum()
        # Kff = self.kern.K(self.X)
        # Qff = tf.matmul(Kuf, LinvKuf, transpose_a=True)

        # Alternative bound on max eigenval:
        # c = tf.reduce_max(tf.reduce_sum(tf.abs(Kff - Qff), 0))
        corrected_noise = self.likelihood.variance.get() + c

        const = -0.5 * num_data * torch.log(2 * float(numpy.pi) * self.likelihood.variance.get())
        logdet = torch.diag(L).log().sum() - torch.diag(LB).log().sum()

        LC = torch.potrf(Kuu + corrected_noise ** -1.0 * torch.matmul(Kuf, Kuf.t()), upper=False)
        v, _ = torch.gesv(corrected_noise ** -1.0 * torch.matmul(Kuf, self.Y), LC)
        quad = -0.5 * corrected_noise ** -1.0 * (self.Y ** 2.0).sum() + 0.5 * (v ** 2.0).sum()

        return const + logdet + quad


class SGPR(GPModel, SGPRUpperMixin):
    """
    Sparse Variational GP regression. The key reference is

    ::

      @inproceedings{titsias2009variational,
        title={Variational learning of inducing variables in
               sparse Gaussian processes},
        author={Titsias, Michalis K},
        booktitle={International Conference on
                   Artificial Intelligence and Statistics},
        pages={567--574},
        year={2009}
      }



    """

    def __init__(self, X, Y, kern, Z, mean_function=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate GPflow objects

        This method only works with a Gaussian likelihood.
        """
        likelihood = likelihoods.Gaussian(ttype=type(X.data))
        super(SGPR,self).__init__(X, Y, kern, likelihood, mean_function, **kwargs)
        self.Z = parameter.Param(Z)
        self.num_data = X.size(0)
        self.num_latent = Y.size(1)

    def compute_log_likelihood(self):
        """
        For a derivation of the terms in here, see the associated
        SGPR notebook.
        """

        num_inducing = self.Z.size(0)
        num_data = self.Y.size(0)
        output_dim = self.Y.size(1)

        err = self.Y - self.mean_function(self.X)
        Kdiag = self.kern.Kdiag(self.X)
        Kuf = self.kern.K(self.Z.get(), self.X)
        jitter = Variable(torch.eye(num_inducing, out=self.Z.data.new())) * self.jitter_level
        Kuu = self.kern.K(self.Z.get()) + jitter
        L = torch.potrf(Kuu, upper=False)
        sigma = self.likelihood.variance.get()**0.5

        # Compute intermediate matrices
        A = torch.gesv(Kuf, L)[0] / sigma  # could use triangular solve
        AAT = torch.matmul(A, A.t())
        B = AAT + Variable(torch.eye(num_inducing, out=AAT.data.new()))
        LB = torch.potrf(B, upper=False)
        Aerr = torch.matmul(A, err)
        c = torch.gesv(Aerr, LB)[0] / sigma # could use triangular solve

        # compute log marginal bound
        bound = -0.5 * num_data * output_dim * float(numpy.log(2 * numpy.pi))
        bound += -output_dim * torch.sum(torch.log(torch.diag(LB)))
        bound = bound - 0.5 * num_data * output_dim * torch.log(self.likelihood.variance.get())
        bound += -0.5 * torch.sum(err**2) / self.likelihood.variance.get()
        bound += 0.5 * torch.sum(c**2)
        bound += -0.5 * output_dim * torch.sum(Kdiag) / self.likelihood.variance.get()
        bound += 0.5 * output_dim * torch.sum(torch.diag(AAT))

        return bound

    def predict_f(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        num_inducing = self.Z.size(0)
        err = self.Y - self.mean_function(self.X)
        Kuf = self.kern.K(self.Z.get(), self.X)
        jitter = Variable(torch.eye(num_inducing, out=self.Z.data.new())) * self.jitter_level
        Kuu = self.kern.K(self.Z.get()) + jitter
        Kus = self.kern.K(self.Z.get(), Xnew)
        sigma = self.likelihood.variance.get()**0.5
        L = torch.potrf(Kuu, upper=False)
        A = torch.gesv(Kuf, L)[0] / sigma # could use triangular solve here and below
        B = torch.matmul(A,A.t()) + Variable(torch.eye(num_inducing, out=A.data.new()))
        LB = torch.potrf(B, upper=False)
        Aerr = torch.matmul(A, err)
        c = torch.gesv(Aerr, LB)[0] / sigma
        tmp1,_ = torch.gesv(Kus, L)
        tmp2,_ = torch.gesv(tmp1,LB)
        mean = torch.matmul(tmp2.t(), c)
        if full_cov:
            var = self.kern.K(Xnew) + torch.matmul(tmp2.t(), tmp2) - torch.matmul(tmp1.t(), tmp1)
            var = var.unsqueeze(2).expand(-1, -1, self.Y.size(1))
        else:
            var = self.kern.Kdiag(Xnew) + (tmp2**2).sum(0) - (tmp1**2).sum(0)
            var = var.unsqueeze(1).expand(-1, self.Y.size(1))
        return mean + self.mean_function(Xnew), var


class GPRFITC(GPModel, SGPRUpperMixin):
    def __init__(self, X, Y, kern, Z, mean_function=None, **kwargs): # was mean_function = Zero()
        """
        This implements GP regression with the FITC approximation.
        The key reference is

        @inproceedings{Snelson06sparsegaussian,
        author = {Edward Snelson and Zoubin Ghahramani},
        title = {Sparse Gaussian Processes using Pseudo-inputs},
        booktitle = {Advances In Neural Information Processing Systems },
        year = {2006},
        pages = {1257--1264},
        publisher = {MIT press}
        }

        Implementation loosely based on code from GPML matlab library although
        obviously gradients are automatic in GPflow.

        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate GPflow objects

        This method only works with a Gaussian likelihood.

        """
        likelihood = likelihoods.Gaussian(ttype=type(X.data))
        super(SGPR,self).__init__(X, Y, kern, likelihood, mean_function, **kwargs)
        self.Z = parameter.Param(Z)
        self.num_data = X.size(0)
        self.num_latent = Y.size(1)

    def _common_terms(self):
        num_inducing = self.Z.size(0)
        err = self.Y - self.mean_function(self.X)  # size N x R
        Kdiag = self.kern.Kdiag(self.X)
        Kuf = self.kern.K(self.Z.get(), self.X)
        jitter = Variable(torch.eye(num_inducing, out=self.Z.data.new())) * self.jitter_level
        Kuu = self.kern.K(self.Z.get()) + jitter

        Luu = torch.potrf(Kuu, upper=False)  # => Luu Luu^T = Kuu
        V, _ = torch.gesv(Kuf, Luu)  # => V^T V = Qff = Kuf^T Kuu^-1 Kuf

        diagQff = (V**2).sum(0)
        nu = Kdiag - diagQff + self.likelihood.variance.get()

        B = torch.eye(num_inducing, out=V.data.new()) + torch.matmul(V / nu, V.t())
        L = torch.potrf(B, upper=False)
        beta = err / nu.unsqueeze(1)  # size N x R
        alpha = torch.matmul(V, beta)  # size N x R

        gamma, _ = torch.gesv(alpha, L)  # size N x R

        return err, nu, Luu, L, alpha, beta, gamma

    def compute_log_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """

        # FITC approximation to the log marginal likelihood is
        # log ( normal( y | mean, K_fitc ) )
        # where K_fitc = Qff + diag( \nu )
        # where Qff = Kfu Kuu^{-1} Kuf
        # with \nu_i = Kff_{i,i} - Qff_{i,i} + \sigma^2

        # We need to compute the Mahalanobis term -0.5* err^T K_fitc^{-1} err
        # (summed over functions).

        # We need to deal with the matrix inverse term.
        # K_fitc^{-1} = ( Qff + \diag( \nu ) )^{-1}
        #            = ( V^T V + \diag( \nu ) )^{-1}
        # Applying the Woodbury identity we obtain
        #            = \diag( \nu^{-1} ) - \diag( \nu^{-1} ) V^T ( I + V \diag( \nu^{-1} ) V^T )^{-1) V \diag(\nu^{-1} )
        # Let \beta =  \diag( \nu^{-1} ) err
        # and let \alpha = V \beta
        # then Mahalanobis term = -0.5* ( \beta^T err - \alpha^T Solve( I + V \diag( \nu^{-1} ) V^T, alpha ) )

        err, nu, Luu, L, alpha, beta, gamma = self._common_terms()

        mahalanobisTerm = -0.5 * (err**2 / nu.unsqueeze(1)).sum() + 0.5 * (gamma**2).sum()

        # We need to compute the log normalizing term -N/2 \log 2 pi - 0.5 \log \det( K_fitc )

        # We need to deal with the log determinant term.
        # \log \det( K_fitc ) = \log \det( Qff + \diag( \nu ) )
        #                    = \log \det( V^T V + \diag( \nu ) )
        # Applying the determinant lemma we obtain
        #                    = \log [ \det \diag( \nu ) \det( I + V \diag( \nu^{-1} ) V^T ) ]
        #                    = \log [ \det \diag( \nu ) ] + \log [ \det( I + V \diag( \nu^{-1} ) V^T ) ]

        constantTerm = -0.5 * self.num_data * float(2*numpy.pi)
        logDeterminantTerm = -0.5 * nu.log().sum() - torch.diag(L).log().sum()
        logNormalizingTerm = constantTerm + logDeterminantTerm

        return mahalanobisTerm + logNormalizingTerm * self.num_latent

    def _build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew.
        """
        _, _, Luu, L, _, _, gamma = self._common_terms()
        Kus = self.kern.K(self.Z.get(), Xnew)  # size  M x Xnew

        w, _ = torch.gesv(Kus, Luu)  # size M x Xnew

        tmp, _ = torch.gesv(gamma, L.t())
        mean = torch.matmul(w.t(), tmp) + self.mean_function(Xnew)
        intermediateA, _ = torch.gesv(w, L)

        if full_cov:
            var = self.kern.K(Xnew) - torch.matmul(w.t(), w) + torch.matmul(intermediateA.t(), intermediateA)
            var = torch.unsqueeze(2).expand(-1, -1, self.Y.size(1))
        else:
            var = self.kern.Kdiag(Xnew) - (w**2).sum(0) + (intermediateA**2).sum(0)  # size Xnew,
            var = torch.unsuqeeze(2).expand(-1, self.Y.size(1))

        return mean, var
