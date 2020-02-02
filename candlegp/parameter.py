# Copyright 2016 James Hensman, Mark van der Wilk,
#                Valentine Svensson, alexggmatthews,
#                PabloLeon, fujiisoup
# Copyright 2017 Artem Artemev @awav
# Copyright 2017 Thomas Viehmann
#
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
import abc
import torch

class ParamWithPrior(torch.nn.Parameter):
    @abc.abstractmethod
    def get(self):
        pass
    @abc.abstractmethod
    def log_jacobian_tensor(self):
        pass
    @abc.abstractstaticmethod
    def untransform(t, out=None):
        pass
    def __init__(self, val, prior=None, dtype=torch.float32):
        pass
    def __new__(cls, val, prior=None, dtype=torch.float32):
        if numpy.isscalar(val):
            val = torch.tensor([val], dtype=dtype)
        raw = cls.untransform(val)
        obj = super(ParamWithPrior, cls).__new__(cls, raw)
        obj.prior = prior
        return obj
    def set(self, t):
        if numpy.isscalar(t):
            t = torch.tensor(t, dtype=self.dtype)
        self.untransform(t, out=self)
    def get_prior(self):
        if self.prior is None:
            return 0.0
        
        log_jacobian = self.log_jacobian() #(unconstrained_tensor)
        logp_var = self.prior.logp(self.get())
        return log_jacobian+logp_var

class PositiveParam(ParamWithPrior): # log(1+exp(r))
    @staticmethod
    def untransform(t, out=None):
        with torch.no_grad():
            return torch.log(torch.exp(t) - 1, out=out)
    def get(self):
        return torch.log(1 + torch.exp(self))
    def log_jacobian(self):
        return -(torch.nn.functional.softplus(-self))

class Param(ParamWithPrior): # unconstrained / untransformed
    @staticmethod
    def untransform(t, out=None):
        if out is None:
            return t
        else:
            return out.copy_(t)
    def get(self):
        return self
    def log_jacobian(self):
        return torch.zeros((), dtype=self.dtype)  # dimension?


class LowerTriangularParam(ParamWithPrior):
    """
    A transform of the form

       tri_mat = vec_to_tri(x)

    x is a free variable, y is always a list of lower triangular matrices sized
    (N x N x D).
    """
    @staticmethod
    def untransform(t, out=None):
        ii,jj = numpy.tril_indices(t.size(0))
        return t[ii,jj]
    def get(self):
        numel = self.size(0)
        N = int((2*numel+0.25)**0.5-0.5)
        ii,jj = numpy.tril_indices(N)
        if self.dim()==2:
            mat = torch.zeros(N,N, self.size(1), dtype=self.dtype)
        else:
            mat = torch.zeros(N, N, dtype=self.dtype)
        mat[ii,jj] = self
        return mat
    def log_jacobian(self):
        return torch.zeros((), dtype=self.dtype)  # dimension?
