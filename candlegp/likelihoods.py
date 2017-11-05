# Copyright 2016 Valentine Svensson, James Hensman, alexggmatthews, Alexis Boukouvalas
# Copyright 2017 Artem Artemev @awav
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
from . import parameter
from . import quadrature
from . import densities

class Likelihood(torch.nn.Module):
    def __init__(self, name=None):
        super(Likelihood, self).__init__()
        self.name = name
        self.num_gauss_hermite_points = 20

    def predict_mean_and_var(self, Fmu, Fvar):
        """
        Given a Normal distribution for the latent function,
        return the mean of Y

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive mean

           \int\int y p(y|f)q(f) df dy

        and the predictive variance

           \int\int y^2 p(y|f)q(f) df dy  - [ \int\int y^2 p(y|f)q(f) df dy ]^2

        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.
        """
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        gh_w /= float(numpy.pi**0.5)
        gh_w = gh_w.reshape(-1, 1)
        shape = Fmu.size()
        Fmu = Fmu.view(-1,1)
        Fvar = Fvar.view(-1,1)
        X = gh_x[None, :] * (2.0 * Fvar)**0.5 + Fmu

        # here's the quadrature for the mean
        E_y = torch.matmul(self.conditional_mean(X), gh_w).view(shape)

        # here's the quadrature for the variance
        integrand = self.conditional_variance(X) + (self.conditional_mean(X))**2
        V_y = torch.matmul(integrand, gh_w).view(shape) - E_y**2

        return E_y, V_y

    def predict_density(self, Fmu, Fvar, Y):
        """
        Given a Normal distribution for the latent function, and a datum Y,
        compute the (log) predictive density of Y.

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive density

           \int p(y=Y|f)q(f) df

        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        gh_x, gh_w = quadrature.hermgauss(self.num_gauss_hermite_points, ttype=type(Fmu.data))

        gh_w = gh_w.reshape(-1, 1) / float(numpy.sqrt(numpy.pi))
        shape = Fmu.size()
        Fmu, Fvar, Y = [e.view(-1, 1) for e in (Fmu, Fvar, Y)]
        X = gh_x * (2.0 * Fvar)**0.5 + Fmu
        Y = Y.expand(-1, self.num_gauss_hermite_points)  # broadcast Y to match X
        logp = self.logp(X, Y)
        return torch.matmul(logp.exp(), gh_w).view(*shape)

    def variational_expectations(self, Fmu, Fvar, Y):
        """
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           \int (\log p(y|f)) q(f) df.


        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """

        gh_x, gh_w = quadrature.hermgauss(self.num_gauss_hermite_points, ttype=type(Fmu.data))
        gh_x = gh_x.view(1, -1)
        gh_w = gh_w.view(-1, 1) / float(numpy.pi)**0.5
        shape = Fmu.size()
        Fmu, Fvar, Y = [e.view(-1, 1) for e in (Fmu, Fvar, Y)]
        X = gh_x * (2.0 * Fvar)**0.5 + Fmu
        Y = Y.expand(-1, self.num_gauss_hermite_points)  # broadcast Y to match X
        logp = self.logp(X, Y)
        return torch.matmul(logp, gh_w).view(*shape)

    def _check_targets(self, Y_np):  # pylint: disable=R0201
        """
        Check that the Y values are valid for the likelihood.
        Y_np is a numpy array.

        The base class check is that the array has two dimensions
        and consists only of floats. The float requirement is so that AutoFlow
        can work with Model.predict_density.
        """
        if not Y.dim() == 2:
            raise ValueError('targets must be shape N x D')
        #if np.array(list(Y_np)).dtype != settings.np_float:
        #    raise ValueError('use {}, even for discrete variables'.format(settings.np_float))


class Gaussian(Likelihood):
    def __init__(self, ttype=torch.FloatTensor):
        Likelihood.__init__(self)
        self.variance = parameter.PositiveParam(1.0, ttype=ttype)

    def logp(self, F, Y):
        return densities.gaussian(F, Y, self.variance.get())

    def conditional_mean(self, F):
        return F

    def conditional_variance(self, F):
        return self.variance.get().expand_as(F)

    def predict_mean_and_var(self, Fmu, Fvar):
        return Fmu, Fvar + self.variance.get()

    def predict_density(self, Fmu, Fvar, Y):
        return densities.gaussian(Fmu, Y, Fvar + self.variance.get())

    def variational_expectations(self, Fmu, Fvar, Y):
        return (-0.5 * numpy.log(2 * numpy.pi) - 0.5 * torch.log(self.variance.get())
                - 0.5 * ((Y - Fmu)**2 + Fvar) / self.variance.get())


def probit(x):
    return 0.5 * (1.0 + torch.erf(x / (2.0**0.5))) * (1 - 2e-3) + 1e-3


class Bernoulli(Likelihood):
    def __init__(self, invlink=probit):
        super(Bernoulli, self).__init__()
        self.invlink = invlink

    def _check_targets(self, Y_np):
        super(Bernoulli, self)._check_targets(Y_np)
        Y_set = set(Y_np.flatten())
        if len(Y_set) > 2 or len(Y_set - set([1.])) > 1:
            raise Warning('all bernoulli variables should be in {1., k}, for some k')

    def logp(self, F, Y):
        return densities.bernoulli(self.invlink(F), Y)

    def predict_mean_and_var(self, Fmu, Fvar):
        if self.invlink is probit:
            p = probit(Fmu / (1 + Fvar)**0.5)
            return p, p - p**2
        else:
            # for other invlink, use quadrature
            return Likelihood.predict_mean_and_var(self, Fmu, Fvar)

    def predict_density(self, Fmu, Fvar, Y):
        p = self.predict_mean_and_var(Fmu, Fvar)[0]
        return densities.bernoulli(p, Y)

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        p = self.invlink(F)
        return p - p**2

    
class Exponential(Likelihood):
    def __init__(self, invlink=torch.exp):
        Likelihood.__init__(self)
        self.invlink = invlink

    def _check_targets(self, Y):
        super(Exponential, self)._check_targets(Y)
        if (Y < 0).any():
            raise ValueError('exponential variables must be positive')

    def logp(self, F, Y):
        return densities.exponential(self.invlink(F), Y)

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        return (self.invlink(F))**2

    def variational_expectations(self, Fmu, Fvar, Y):
        if self.invlink is torch.exp:
            return - torch.exp(-Fmu + Fvar / 2) * Y - Fmu
        return super(Exponential, self).variational_expectations(Fmu, Fvar, Y)

