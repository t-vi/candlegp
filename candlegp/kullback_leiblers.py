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
import numpy

def batch_tril(A):
    B = A.clone()
    ii,jj = numpy.triu_indices(B.size(-2), k=1, m=B.size(-1))
    B[...,ii,jj] = 0
    return B

def batch_diag(A):
    ii,jj = numpy.diag_indices(min(A.size(-2),A.size(-1)))
    return A[...,ii,jj]

def gauss_kl_white(q_mu, q_sqrt):
    """
    Compute the KL divergence from

          q(x) = N(q_mu, q_sqrt^2)
    to
          p(x) = N(0, I)

    We assume multiple independent distributions, given by the columns of
    q_mu and the last dimension of q_sqrt.

    q_mu is a matrix, each column contains a mean

    q_sqrt is a 3D tensor, each matrix within is a lower triangular square-root
        matrix of the covariance.
    """
    KL = 0.5 * (q_mu**2).sum()               # Mahalanobis term
    KL += -0.5 * numpy.prod(q_sqrt.size()[1:]) # constant term
    L =  batch_tril(q_sqrt.permute(2,0,1))   # force lower triangle
    KL -= batch_diag(L).log().sum()          # logdet sum log(L**2)
    KL += 0.5 * (L**2).sum()                 # Trace term.
    return KL


def gauss_kl_white_diag(q_mu, q_sqrt):
    """
    Compute the KL divergence from

          q(x) = N(q_mu, q_sqrt^2)
    to
          p(x) = N(0, I)

    We assume multiple independent distributions, given by the columns of
    q_mu and q_sqrt

    q_mu is a matrix, each column contains a mean

    q_sqrt is a matrix, each column represents the diagonal of a square-root
        matrix of the covariance.
    """

    KL = 0.5 * (q_mu**2).sum()                    # Mahalanobis term
    KL += -0.5 * q_sqrt.numel()
    KL = KL - q_sqrt.abs().log().sum()            # Log-det of q-cov
    KL += 0.5 * (q_sqrt**2).sum()                 # Trace term
    return KL


def gauss_kl_diag(q_mu, q_sqrt, K):
    """
    Compute the KL divergence from

          q(x) = N(q_mu, q_sqrt^2)
    to
          p(x) = N(0, K)

    We assume multiple independent distributions, given by the columns of
    q_mu and q_sqrt.

    q_mu is a matrix, each column contains a mean

    q_sqrt is a matrix, each column represents the diagonal of a square-root
        matrix of the covariance of q.

    K is a positive definite matrix: the covariance of p.
    """
    L = torch.potrf(K, upper=False)
    alpha,_ = torch.gesv(q_mu, L)
    KL = 0.5 * (alpha**2).sum()                  # Mahalanobis term.
    num_latent = q_sqrt.size(1)
    KL += num_latent * torch.diag(L).log().sum() # Prior log-det term.
    KL += -0.5 * q_sqrt.numel()                  # constant term
    KL += - q_sqrt.log().sum()                   # Log-det of q-cov
    K_inv,_ = torch.potrs(Variable(torch.eye(L.size(0), out=L.data.new())), L, upper=False) 
    KL += 0.5 * (torch.diag(K_inv).unsqueeze(1) * q_sqrt**2).sum()  # Trace term.
    return KL


def gauss_kl(q_mu, q_sqrt, K):
    """
    Compute the KL divergence from

          q(x) = N(q_mu, q_sqrt^2)
    to
          p(x) = N(0, K)

    We assume multiple independent distributions, given by the columns of
    q_mu and the last dimension of q_sqrt.

    q_mu is a matrix, each column contains a mean.

    q_sqrt is a 3D tensor, each matrix within is a lower triangular square-root
        matrix of the covariance of q.

    K is a positive definite matrix: the covariance of p.
    """
    L = torch.potrf(K, upper=False)
    alpha,_ = torch.gesv(q_mu, L)
    KL = 0.5 * (alpha**2).sum()                  # Mahalanobis term.
    num_latent = q_sqrt.size(2)
    KL += num_latent * torch.tiag(L).log().sum() # Prior log-det term.
    KL += -0.5 * numpy.prod(q_sqrt.size()[1:])  # constant term
    Lq = batch_tril(q_sqrt.permute(2, 0, 1))  # force lower triangle
    KL += batch_diag(Lq).log().sum()         # logdet
    LiLq,_ = torch.gesv(Lq.view(-1,L.size(-1)), L).view(*L.size()) # batch with same LHS
    KL += 0.5 * (LiLq**2).sum()  # Trace term
    return KL
