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
from torch.autograd import Variable

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
    def __new__(cls, val, prior=None):
        if isinstance(val, torch.autograd.Variable):
            val = val.data
        elif numpy.isscalar(val):
            val = torch.FloatTensor([val])
        raw = cls.untransform(val)
        obj = super(ParamWithPrior, cls).__new__(cls, raw)
        obj.prior = prior
        return obj
    def set(self, t):
        if isinstance(t, torch.autograd.Variable):
            t = t.data
        elif numpy.isscalar(t):
            t = torch.FloatTensor([t])
        self.untransform(t, out=self.data)
    def get_prior(self):
        if self.prior is None:
            return 0.0
        
        log_jacobian = self.log_jacobian() #(unconstrained_tensor)
        logp_var = self.prior.logp(self.get())
        return log_jacobian+logp_var

class PositiveParam(ParamWithPrior): # log(1+exp(r))
    @staticmethod
    def untransform(t, out=None):
        return torch.log(torch.exp(t)-1, out=out)
    def get(self):
        return torch.log(1+torch.exp(self))
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
        return Variable(self.data.new(1).zero_()) # dimension?
