# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews
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
import numpy
from . import parameter

class Kern(torch.nn.Module):
    """
    The basic kernel class. Handles input_dim and active dims, and provides a
    generic '_slice' function to implement them.
    """

    def __init__(self, input_dim, active_dims=None, name=None):
        """
        input dim is an integer
        active dims is either an iterable of integers or None.

        Input dim is the number of input dimensions to the kernel. If the
        kernel is computed on a matrix X which has more columns than input_dim,
        then by default, only the first input_dim columns are used. If
        different columns are required, then they may be specified by
        active_dims.

        If active dims is None, it effectively defaults to range(input_dim),
        but we store it as a slice for efficiency.
        """
        super(Kern, self).__init__()
        self.name = name
        self.input_dim = int(input_dim)
        if active_dims is None:
            self.active_dims = slice(input_dim)
        elif isinstance(active_dims, slice):
            self.active_dims = active_dims
            if active_dims.start is not None and active_dims.stop is not None and active_dims.step is not None:
                assert len(range(*active_dims)) == input_dim  # pragma: no cover
        else:
            self.active_dims = numpy.array(active_dims, dtype=numpy.int32)
            assert len(active_dims) == input_dim

        self.num_gauss_hermite_points = 20

    def eKdiag(self, Xmu, Xcov):
        """
        Computes <K_xx>_q(x).
        :param Xmu: Mean (NxD)
        :param Xcov: Covariance (NxDxD or NxD)
        :return: (N)
        """
        self._check_quadrature()
        Xmu, _ = self._slice(Xmu, None)
        Xcov = self._slice_cov(Xcov)
        return mvnquad(lambda x: self.Kdiag(x, presliced=True),
                       Xmu, Xcov,
                       self.num_gauss_hermite_points, self.input_dim)  # N

    def eKxz(self, Z, Xmu, Xcov):
        """
        Computes <K_xz>_q(x) using quadrature.
        :param Z: Fixed inputs (MxD).
        :param Xmu: X means (NxD).
        :param Xcov: X covariances (NxDxD or NxD).
        :return: (NxM)
        """
        self._check_quadrature()
        Xmu, Z = self._slice(Xmu, Z)
        Xcov = self._slice_cov(Xcov)
        M = tf.shape(Z)[0]
        return mvnquad(lambda x: self.K(x, Z, presliced=True), Xmu, Xcov, self.num_gauss_hermite_points,
                       self.input_dim, Dout=(M,))  # (H**DxNxD, H**D)

    def exKxz(self, Z, Xmu, Xcov):
        """
        Computes <x_{t-1} K_{x_t z}>_q(x) for each pair of consecutive X's in
        Xmu & Xcov.
        :param Z: Fixed inputs (MxD).
        :param Xmu: X means (T+1xD).
        :param Xcov: 2xT+1xDxD. [0, t, :, :] contains covariances for x_t. [1, t, :, :] contains the cross covariances
        for t and t+1.
        :return: (TxMxD).
        """
        self._check_quadrature()
        # Slicing is NOT needed here. The desired behaviour is to *still* return an NxMxD matrix. As even when the
        # kernel does not depend on certain inputs, the output matrix will still contain the outer product between the
        # mean of x_{t-1} and K_{x_t Z}. The code here will do this correctly automatically, since the quadrature will
        # still be done over the distribution x_{t-1, t}, only now the kernel will not depend on certain inputs.
        # However, this does mean that at the time of running this function we need to know the input *size* of Xmu, not
        # just `input_dim`.
        M = tf.shape(Z)[0]
        D = self.input_size if hasattr(self, 'input_size') else self.input_dim  # Number of actual input dimensions
        assert Xmu.size(1)==D

        # First, transform the compact representation of Xmu and Xcov into a
        # list of full distributions.
        fXmu = tf.concat((Xmu[:-1, :], Xmu[1:, :]), 1)  # Nx2D
        fXcovt = tf.concat((Xcov[0, :-1, :, :], Xcov[1, :-1, :, :]), 2)  # NxDx2D
        fXcovb = tf.concat((tf.transpose(Xcov[1, :-1, :, :], (0, 2, 1)), Xcov[0, 1:, :, :]), 2)
        fXcov = tf.concat((fXcovt, fXcovb), 1)
        return mvnquad(lambda x: self.K(x[:, :D], Z).unsqueeze(2) *
                                 x[:, D:].unsqueeze(1),
                       fXmu, fXcov, self.num_gauss_hermite_points,
                       2 * D, Dout=(M, D))

    def eKzxKxz(self, Z, Xmu, Xcov):
        """
        Computes <K_zx Kxz>_q(x).
        :param Z: Fixed inputs MxD.
        :param Xmu: X means (NxD).
        :param Xcov: X covariances (NxDxD or NxD).
        :return: NxMxM
        """
        self._check_quadrature()
        Xmu, Z = self._slice(Xmu, Z)
        Xcov = self._slice_cov(Xcov)
        M = Z.size(0)

        def KzxKxz(x):
            Kxz = self.K(x, Z, presliced=True)
            return tf.expand_dims(Kxz, 2) * tf.expand_dims(Kxz, 1)

        return mvnquad(KzxKxz,
                       Xmu, Xcov, self.num_gauss_hermite_points,
                       self.input_dim, Dout=(M, M))

    def _check_quadrature(self):
        if settings.numerics.ekern_quadrature == "warn":
            warnings.warn("Using numerical quadrature for kernel expectation of %s. Use gpflow.ekernels instead." %
                          str(type(self)))
        if settings.numerics.ekern_quadrature == "error" or self.num_gauss_hermite_points == 0:
            raise RuntimeError("Settings indicate that quadrature may not be used.")

    def _slice(self, X, X2):
        """
        Slice the correct dimensions for use in the kernel, as indicated by
        `self.active_dims`.
        :param X: Input 1 (NxD).
        :param X2: Input 2 (MxD), may be None.
        :return: Sliced X, X2, (Nxself.input_dim).
        """
        #if isinstance(self.active_dims, slice):
        X = X[:, self.active_dims]
        if X2 is not None:
            X2 = X2[:, self.active_dims]
        # I think advanced indexing does the right thing also for the second case
        #else:
        assert X.size(1)==self.input_dim
        return X, X2

    def _slice_cov(self, cov):
        """
        Slice the correct dimensions for use in the kernel, as indicated by
        `self.active_dims` for covariance matrices. This requires slicing the
        rows *and* columns. This will also turn flattened diagonal
        matrices into a tensor of full diagonal matrices.
        :param cov: Tensor of covariance matrices (NxDxD or NxD).
        :return: N x self.input_dim x self.input_dim.
        """
        cov = tf.cond(tf.equal(tf.rank(cov), 2), lambda: tf.matrix_diag(cov), lambda: cov)

        if isinstance(self.active_dims, slice):
            cov = cov[..., self.active_dims, self.active_dims]
        else:
            cov_shape = cov.size()
            covr = cov.view(-1, cov_shape[-1], cov_shape[-1])
            gather1 = torch.gather(tf.transpose(covr, [2, 1, 0]), self.active_dims)
            gather2 = torch.gather(tf.transpose(gather1, [1, 0, 2]), self.active_dims)
            cov = tf.reshape(tf.transpose(gather2, [2, 0, 1]),
                             tf.concat([cov_shape[:-2], [len(self.active_dims), len(self.active_dims)]], 0))
        return cov

    def __add__(self, other):
        return Add([self, other])

    def __mul__(self, other):
        return Prod([self, other])


class Static(Kern):
    """
    Kernels who don't depend on the value of the inputs are 'Static'.  The only
    parameter is a variance.
    """

    def __init__(self, input_dim, variance=1.0, active_dims=None, name=None):
        super(Static, self).__init__(input_dim, active_dims, name=name)
        self.variance = parameter.PositiveParam(variance)

    def Kdiag(self, X, presliced=False):
        return self.variance.get().expand(X.size(0))


class White(Static):
    """
    The White kernel
    """

    def K(self, X, X2=None, presliced=False):
        if X2 is None:
            d = self.variance.get().expand(X.size(0))
            return torch.diag(d)
        else:
            shape = tf.stack([tf.shape(X)[0], tf.shape(X2)[0]])
            return Variable(X.data.new(X.size(0),X2.size(0)).zero_())


class Constant(Static):
    """
    The Constant (aka Bias) kernel
    """

    def K(self, X, X2=None, presliced=False):
        if X2 is None:
            return self.variance.get().expand(X.size(0), X.size(0))
        else:
            return self.variance.get().expand(X.size(0), X2.size(0))

class Bias(Constant):
    """
    Another name for the Constant kernel, included for convenience.
    """
    pass


class Stationary(Kern):
    """
    Base class for kernels that are stationary, that is, they only depend on

        r = || x - x' ||

    This class handles 'ARD' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(self, input_dim, variance=1.0, lengthscales=None,
                 active_dims=None, ARD=False, name=None):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter
        - lengthscales is the initial value for the lengthscales parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - ARD specifies whether the kernel has one lengthscale per dimension
          (ARD=True) or a single lengthscale (ARD=False).
        """
        super(Stationary, self).__init__(input_dim, active_dims, name=name)
        self.variance = parameter.PositiveParam(variance)
        if ARD:
            if lengthscales is None:
                lengthscales = torch.ones(input_dim)
            else:
                # accepts float or array:
                lengthscales = lengthscales * torch.ones(input_dim)
            self.lengthscales = parameter.PositiveParam(lengthscales)
            self.ARD = True
        else:
            if lengthscales is None:
                lengthscales = 1.0
            self.lengthscales = parameter.PositiveParam(lengthscales)
            self.ARD = False

    def square_dist(self, X, X2):
        X = X / self.lengthscales.get()
        Xs = (X**2).sum(1)

        if X2 is None:
            dist = -2 * torch.matmul(X, X.t())
            dist += Xs.view(-1, 1) + Xs.view(1, -1)
            return dist

        X2 = X2 / self.lengthscales.get()
        X2s = (X2**2).sum(1)
        dist = -2 * torch.matmul(X, X2.t())
        dist += Xs.view(-1, 1) + X2s.view(1, -1)
        return dist


    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return (r2 + 1e-12)**0.5

    def Kdiag(self, X, presliced=False):
        return self.variance.get().expand(X.size(0))

class RBF(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        res = self.variance.get() * torch.exp(-0.5 * self.square_dist(X, X2))
        return res

class Exponential(Stationary):
    """
    The Exponential kernel
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance.get() * torch.exp(-0.5 * r)


class Matern12(Stationary):
    """
    The Matern 1/2 kernel
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance.get() * torch.exp(-r)


class Matern32(Stationary):
    """
    The Matern 3/2 kernel
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance.get() * (1. + (3.**0.5) * r) * torch.exp(-(3.**0.5) * r)


class Matern52(Stationary):
    """
    The Matern 5/2 kernel
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance.get() * (1.0 + (5.**0.5) * r + 5. / 3. * r**2) * torch.exp(-(5.**0.5) * r)


class Cosine(Stationary):
    """
    The Cosine kernel
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance.get() * torch.cos(r)


class ArcCosine(Kern):
    """
    The Arc-cosine family of kernels which mimics the computation in neural
    networks. The order parameter specifies the assumed activation function.
    The Multi Layer Perceptron (MLP) kernel is closely related to the ArcCosine
    kernel of order 0. The key reference is

    ::

        @incollection{NIPS2009_3628,
            title = {Kernel Methods for Deep Learning},
            author = {Youngmin Cho and Lawrence K. Saul},
            booktitle = {Advances in Neural Information Processing Systems 22},
            year = {2009},
            url = {http://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning.pdf}
        }
    """

    implemented_orders = {0, 1, 2}
    def __init__(self, input_dim,
                 order=0,
                 variance=1.0, weight_variances=1., bias_variance=1.,
                 active_dims=None, ARD=False, name=None):
        """
        - input_dim is the dimension of the input to the kernel
        - order specifies the activation function of the neural network
          the function is a rectified monomial of the chosen order.
        - variance is the initial value for the variance parameter
        - weight_variances is the initial value for the weight_variances parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - bias_variance is the initial value for the bias_variance parameter
          defaults to 1.0.
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - ARD specifies whether the kernel has one weight_variance per dimension
          (ARD=True) or a single weight_variance (ARD=False).
        """
        super(ArcCosine, self).__init__(input_dim, active_dims, name=name)

        if order not in self.implemented_orders:
            raise ValueError('Requested kernel order is not implemented.')
        self.order = order

        self.variance = parameter.PositiveParam(variance)
        self.bias_variance = parameter.PositiveParam(variance)
        self.ARD = ARD
        if ARD:
            if weight_variances is None:
                weight_variances = self.variance.data.new(input_dim).fill_(1.0)
            else:
                # accepts float or Tensor:
                weight_variances = weight_variances * self.variance.data.new(input_dim).fill_(1.0)
            self.weight_variances = parameter.PositiveParam(weight_variances)
        else:
            if weight_variances is None:
                weight_variances = 1.0
            self.weight_variances = parameter.PositiveParam(weight_variances)

    def _weighted_product(self, X, X2=None):
        if X2 is None:
            return (self.weight_variances.get() * (X**2)).sum(1) + self.bias_variance.get()
        return torch.matmul(self.weight_variances.get() * X, X2.t()) + self.bias_variance.get()

    def _J(self, theta):
        """
        Implements the order dependent family of functions defined in equations
        4 to 7 in the reference paper.
        """
        if self.order == 0:
            return float(numpy.pi) - theta
        elif self.order == 1:
            return torch.sin(theta) + (float(numpy.pi) - theta) * torch.cos(theta)
        elif self.order == 2:
            return 3. * torch.sin(theta) * torch.cos(theta) + (float(numpy.pi) - theta) * (1. + 2. * torch.cos(theta) ** 2)

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)

        X_denominator = self._weighted_product(X)**0.5
        if X2 is None:
            X2 = X
            X2_denominator = X_denominator
        else:
            X2_denominator = self._weighted_product(X2)**0.5

        numerator = self._weighted_product(X, X2)
        cos_theta = numerator / X_denominator[:, None] / X2_denominator[None, :]
        jitter = 1e-15
        theta = torch.acos(jitter + (1 - 2 * jitter) * cos_theta)

        return ( self.variance.get() * (1. / float(numpy.pi)) * self._J(theta)
                *X_denominator[:, None] ** self.order
                *X2_denominator[None, :] ** self.order)

    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)

        X_product = self._weighted_product(X)
        theta = 0
        return self.variance.get() * (1. / float(numpy.pi)) * self._J(theta) * X_product ** self.order


class Combination(Kern):
    """
    Combine  a list of kernels, e.g. by adding or multiplying (see inheriting
    classes).

    The names of the kernels to be combined are generated from their class
    names.
    """

    def __init__(self, kern_list, name=None):
        for k in kern_list:
            assert isinstance(k, Kern), "can only add/multiply Kern instances"

        input_dim = numpy.max([k.input_dim if type(k.active_dims) is slice else numpy.max(k.active_dims) + 1 for k in kern_list])
        super(Combination, self).__init__(input_dim=input_dim, name=name)

        # add kernels to a list, flattening out instances of this class therein
        self.kern_list = torch.nn.ModuleList()
        for k in kern_list:
            if isinstance(k, self.__class__):
                self.kern_list.extend(k.kern_list)
            else:
                self.kern_list.append(k)

    @property
    def on_separate_dimensions(self):
        """
        Checks whether the kernels in the combination act on disjoint subsets
        of dimensions. Currently, it is hard to asses whether two slice objects
        will overlap, so this will always return False.
        :return: Boolean indicator.
        """
        if numpy.any([isinstance(k.active_dims, slice) for k in self.kern_list]):
            # Be conservative in the case of a slice object
            return False
        else:
            dimlist = [k.active_dims for k in self.kern_list]
            overlapping = False
            for i, dims_i in enumerate(dimlist):
                for dims_j in dimlist[i + 1:]:
                    if numpy.any(dims_i.reshape(-1, 1) == dims_j.reshape(1, -1)):
                        overlapping = True
            return not overlapping

        
class Add(Combination):
    def K(self, X, X2=None, presliced=False):
        res = 0.0
        for k in self.kern_list:
            res += k.K(X, X2, presliced=presliced)
        return res

    def Kdiag(self, X, presliced=False):
        res = 0.0
        for k in self.kern_list:
            res += k.Kdiag(X, presliced=presliced)
        return res


class Prod(Combination):
    def K(self, X, X2=None, presliced=False):
        res = 1.0
        for k in self.kern_list:
            res *= k.K(X, X2, presliced=presliced)
        return res

    def Kdiag(self, X, presliced=False):
        res = 1.0
        for k in self.kern_list:
            res *= k.Kdiag(X, presliced=presliced)
        return res

