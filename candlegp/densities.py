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

import numpy
import scipy.special
import torch
from torch.autograd import Variable

def gammaln(x):
    # attention: Not differentiable!
    if numpy.isscalar(x):
        y = float(scipy.special.gammaln(x))
    elif isinstance(x,(torch.Tensor, torch.DoubleTensor)):
        y = type(x)(scipy.special.gammaln(x.numpy()))
    elif isinstance(x,Variable):
        y = Variable(type(x.data)(scipy.special.gammaln(x.data.numpy())))
    else:
        raise ValueError("Unsupported input type "+str(type(x)))
    return y
    
    
def gaussian(x, mu, var):
    return -0.5 * (float(numpy.log(2 * numpy.pi)) + torch.log(var) + (mu-x)**2/var)

def lognormal(x, mu, var):
    lnx = torch.log(x)
    return gaussian(lnx, mu, var) - lnx

def bernoulli(p, y):
    return torch.log(y*p+(1-y)*(1-p))

def poisson(lamb, y):
    return y * torch.log(lamb) - lamb - gammaln(y + 1.)

def exponential(lamb, y):
    return - y/lamb - torch.log(lamb)

def gamma(shape, scale, x):
    return (-shape * torch.log(scale) - gammaln(shape)
            + (shape - 1.) * torch.log(x) - x / scale)

def student_t(x, mean, scale, deg_free): # todo
    const = tf.lgamma(tf.cast((deg_free + 1.) * 0.5, float_type))\
            - tf.lgamma(tf.cast(deg_free * 0.5, float_type))\
            - 0.5*(2*tf.log(scale) + tf.cast(tf.log(deg_free), float_type)
                   + np.log(np.pi))
    const = tf.cast(const, float_type)
    return const - 0.5*(deg_free + 1.) * \
           tf.log(1. + (1. / deg_free) * (tf.square((x - mean) / scale)))

def beta(alpha, beta, y):
    # need to clip y, since log of 0 is nan...
    y = torch.clamp(y, min=1e-6, max=1-1e-6)
    return ((alpha - 1.) * torch.log(y) + (beta - 1.) * torch.log(1. - y)
            + gammaln(alpha + beta)
            - gammaln(alpha)
            - gammaln(beta))

def laplace(mu, sigma, y):
    return - torch.abs(mu - y) / sigma - torch.log(2. * sigma)

def multivariate_normal(x, mu, L):
    """
    L is the Cholesky decomposition of the covariance.
    x and mu are either vectors (ndim=1) or matrices. In the matrix case, we
    assume independence over the *columns*: the number of rows must match the
    size of L.
    """
    d = x - mu
    if d.dim()==1:
        d = d.unsqueeze(1)
    alpha,_ = torch.gesv(d, L)
    alpha = alpha.squeeze(1)
    num_col = 1 if x.dim() == 1 else x.size(1)
    num_dims = x.size(0)
    ret = - 0.5 * num_dims * num_col * float(numpy.log(2 * numpy.pi))
    ret += - num_col * torch.diag(L).sum()
    ret += - 0.5 * (alpha**2).sum()
    return ret
