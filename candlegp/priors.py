# Copyright 2016 Valentine Svensson, James Hensman, alexggmatthews
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

from . import densities

def wrap(x, **argd):
    if numpy.isscalar(x):
        x = Variable(torch.Tensor([x]),**argd)
    elif isinstance(x, [torch.Tensor, torch.DoubleTensor]):
        x = Variable(x, **argd)
    return x

class Prior: 
    pass

# these should be ocnverted to use torch.densities
class Gaussian(Prior):
    def __init__(self, mu, var):
        Prior.__init__(self)
        self.mu = wrap(mu)
        self.var = wrap(var)

    def logp(self, x):
        return densities.gaussian(x, self.mu, self.var).sum()

    def sample(self, shape=(1,)):
        return self.mu + (self.var**0.5) * Variable(torch.randn(*shape))

    def __str__(self):
        return "N("+str(self.mu.data.cpu().numpy()) + "," + str(self.var.data.cpu().numpy()) + ")"


class LogNormal(Prior):
    def __init__(self, mu, var):
        Prior.__init__(self)           
        self.mu = wrap(mu)
        self.var = wrap(var)

    def logp(self, x):
        return densities.lognormal(x, self.mu, self.var).sum()

    def sample(self, shape=(1,)):
        return (self.mu + (self.var**0.5) * Variable(torch.randn(*shape))).exp()

    def __str__(self):
        return "logN("+str(self.mu.data.cpu().numpy()) + "," + str(self.var.data.cpu().numpy()) + ")"


class Gamma(Prior):
    def __init__(self, shape, scale):
        Prior.__init__(self)
        self.shape = wrap(shape)
        self.scale = wrap(scale)

    def logp(self, x):
        return densities.gamma(self.shape, self.scale, x).sum()

    def sample(self, shape=(1,)):
        return Variable(torch.Tensor(numpy.random.gamma(self.shape, self.scale, size=shape)))

    def __str__(self):
        return "Ga("+str(self.shape.data.cpu().numpy()) + "," + str(self.scale.data.cpu().numpy()) + ")"


class Laplace(Prior):
    def __init__(self, mu, sigma):
        Prior.__init__(self)
        self.mu = wrap(mu)
        self.sigma = wrap(sigma)

    def logp(self, x):
        return densities.laplace(self.mu, self.sigma, x).sum()

    def sample(self, shape=(1,)):
        return Variable(torch.Tensor(numpy.random.laplace(self.mu, self.sigma, size=shape)))

    def __str__(self):
        return "Lap.("+str(self.mu.data.cpu().numpy()) + "," + str(self.sigma.data.cpu().numpy()) + ")"


class Beta(Prior):
    def __init__(self, a, b):
        Prior.__init__(self)
        self.a = wrap(a)
        self.b = wrap(b)

    def logp(self, x):
        return tf.reduce_sum(densities.beta(self.a, self.b, x))

    def sample(self, shape=(1,)):
        return Variable(torch.Tensor(self.a, self.b, size=shape))

    def __str__(self):
        return "Beta(" + str(self.a.data.cpu().numpy()) + "," + str(self.b.data.cpu().numpy()) + ")"


class Uniform(Prior):
    def __init__(self, lower=0., upper=1.):
        Prior.__init__(self)
        lower = wrap(lower)
        upper = wrap(upper)
        self.log_height = - torch.log(upper - lower)
        self.lower, self.upper = lower, upper

    def logp(self, x):
        assert x.dim()==1
        return self.log_height * x.size(0)

    def sample(self, shape=(1,)):
        return (self.lower +
                (self.upper - self.lower)*torch.rand(*shape))

    def __str__(self):
        return "U("+str(self.lower.data.cpu().numpy()) + "," + str(self.upper.data.cpu().numpy()) + ")"
