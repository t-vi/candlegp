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

class RobustMax(object):
    """
    This class represent a multi-class inverse-link function. Given a vector
    f=[f_1, f_2, ... f_k], the result of the mapping is

    y = [y_1 ... y_k]

    with

    y_i = (1-eps)  i == argmax(f)
          eps/(k-1)  otherwise.


    """

    def __init__(self, num_classes, epsilon=1e-3):
        self.epsilon = epsilon
        self.num_classes = num_classes
        self._eps_K1 = self.epsilon / (self.num_classes - 1.)

    def __call__(self, F):
        _,i = torch.max(F.data, 1)
        one_hot = Variable(F.data.new(F.size(0), self.num_classes).fill_(self._eps_K1).scatter_(1,i,1-self.epsilon))
        return one_hot

    def prob_is_largest(self, Y, mu, var, gh_x, gh_w):
        Y = Y.long()
        # work out what the mean and variance is of the indicated latent function.
        oh_on = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 1., 0.), settings.tf_float)
        mu_selected = tf.reduce_sum(oh_on * mu, 1)
        var_selected = tf.reduce_sum(oh_on * var, 1)

        # generate Gauss Hermite grid
        X = tf.reshape(mu_selected, (-1, 1)) + gh_x * tf.reshape(
            tf.sqrt(tf.clip_by_value(2. * var_selected, 1e-10, np.inf)), (-1, 1))

        # compute the CDF of the Gaussian between the latent functions and the grid (including the selected function)
        dist = (tf.expand_dims(X, 1) - tf.expand_dims(mu, 2)) / tf.expand_dims(
            tf.sqrt(tf.clip_by_value(var, 1e-10, np.inf)), 2)
        cdfs = 0.5 * (1.0 + tf.erf(dist / np.sqrt(2.0)))

        cdfs = cdfs * (1 - 2e-4) + 1e-4

        # blank out all the distances on the selected latent function
        oh_off = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 0., 1.), settings.tf_float)
        cdfs = cdfs * tf.expand_dims(oh_off, 2) + tf.expand_dims(oh_on, 2)

        # take the product over the latent functions, and the sum over the GH grid.
        return tf.matmul(tf.reduce_prod(cdfs, reduction_indices=[1]), tf.reshape(gh_w / np.sqrt(np.pi), (-1, 1)))


class MultiClass(Likelihood):
    def __init__(self, num_classes, invlink=None):
        """
        A likelihood that can do multi-way classification.
        Currently the only valid choice
        of inverse-link function (invlink) is an instance of RobustMax.
        """
        Likelihood.__init__(self)
        self.num_classes = num_classes
        if invlink is None:
            invlink = RobustMax(self.num_classes)
        elif not isinstance(invlink, RobustMax):
            raise NotImplementedError
        self.invlink = invlink

    def _check_targets(self, Y_np):
        super(MultiClass, self)._check_targets(Y_np)
        if not set(Y_np.flatten()).issubset(set(np.arange(self.num_classes))):
            raise ValueError('multiclass likelihood expects inputs to be in {0., 1., 2.,...,k-1}')
        if Y_np.shape[1] != 1:
            raise ValueError('only one dimension currently supported for multiclass likelihood')

    def logp(self, F, Y):
        if isinstance(self.invlink, RobustMax):
            hits = tf.equal(tf.expand_dims(tf.argmax(F, 1), 1), tf.cast(Y, tf.int64))
            yes = tf.ones(tf.shape(Y), dtype=settings.tf_float) - self.invlink.epsilon
            no = tf.zeros(tf.shape(Y), dtype=settings.tf_float) + self.invlink._eps_K1
            p = tf.where(hits, yes, no)
            return tf.log(p)
        else:
            raise NotImplementedError

    def variational_expectations(self, Fmu, Fvar, Y):
        if isinstance(self.invlink, RobustMax):
            gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
            p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
            return p * np.log(1 - self.invlink.epsilon) + (1. - p) * np.log(self.invlink._eps_K1)
        else:
            raise NotImplementedError

    def predict_mean_and_var(self, Fmu, Fvar):
        if isinstance(self.invlink, RobustMax):
            # To compute this, we'll compute the density for each possible output
            possible_outputs = [tf.fill(tf.stack([tf.shape(Fmu)[0], 1]), np.array(i, dtype=np.int64)) for i in
                                range(self.num_classes)]
            ps = [self._predict_non_logged_density(Fmu, Fvar, po) for po in possible_outputs]
            ps = tf.transpose(tf.stack([tf.reshape(p, (-1,)) for p in ps]))
            return ps, ps - tf.square(ps)
        else:
            raise NotImplementedError

    def predict_density(self, Fmu, Fvar, Y):
        return tf.log(self._predict_non_logged_density(Fmu, Fvar, Y))

    def _predict_non_logged_density(self, Fmu, Fvar, Y):
        if isinstance(self.invlink, RobustMax):
            gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
            p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
            return p * (1 - self.invlink.epsilon) + (1. - p) * (self.invlink._eps_K1)
        else:
            raise NotImplementedError

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - tf.square(p)

