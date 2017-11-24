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


import torch
from torch.autograd import Variable
import numpy

def is_finite(x):
    INF = x.data.new(1).fill_(numpy.inf)
    if isinstance(x, Variable):
        return Variable((x.data<INF) & (x.data > -INF))
    else:
        return (x<INF) & (x>-INF)

def hmc_sample(model, num_samples, epsilon, lmin=1, lmax=2, thin=1, burn=0):
    """
    A straight-forward HMC implementation. The mass matrix is assumed to be the
    identity.
    f is a python function that returns the energy and its gradient
      f(x) = E(x), dE(x)/dx
    we then generate samples from the distribution
      pi(x) = exp(-E(x))/Z
    - num_samples is the number of samples to generate.
    - Lmin, Lmax, epsilon are parameters of the HMC procedure to be tuned.
    - x0 is the starting position for the procedure.
    - verbose is a flag which turns on the display of the running accept ratio.
    - thin is an integer which specifies the thinning interval
    - burn is an integer which specifies how many initial samples to discard.
    - RNG is a random number generator
    - return_logprobs is a boolean indicating whether to return the log densities alongside the samples.
    The total number of iterations is given by
      burn + thin * num_samples
    The return shape is always num_samples x D.
    The leapfrog (Verlet) integrator works by picking a random number of steps
    uniformly between Lmin and Lmax, and taking steps of length epsilon.
    """

    def logprob_grads():
        logprob = -model.objective()
        grads = torch.autograd.grad(logprob, [p for p in model.parameters() if p.requires_grad])
        return logprob, grads

    def thinning(thin_iterations, epsilon, lmin, lmax):
        logprob, grads = logprob_grads()
        for i in range(thin):
            xs_prev = [p.data.clone() for p in model.parameters() if p.requires_grad]
            grads_prev = grads
            logprob_prev = logprob
            ps_init = [Variable(xs_prev[0].new(*p.size()).normal_()) for p in model.parameters() if p.requires_grad]
            ps = [p + 0.5 * epsilon * grad for p,grad in zip(ps_init, grads_prev)]

            max_iterations = int((torch.rand(1)*(lmax+1-lmin)+lmin)[0])

            # leapfrog step
            proceed = True
            i_ps = 0
            while proceed and i_ps < max_iterations:
                for x, p in zip([p for p in model.parameters() if p.requires_grad], ps):
                    x.data += epsilon*p.data
                _, grads = logprob_grads()
                proceed = torch.stack([is_finite(grad).prod() for grad in grads], dim=0).prod().data[0]
                if proceed:
                    ps = [p + epsilon * grad for p, grad in zip(ps, grads)]
                i_ps += 1
            logprob, grads = logprob_grads()

            if proceed:
                ps_upd = [p - 0.5 * epsilon * grad for p,grad in zip(ps, grads)]
                log_kinetic_energy = 0
                for p in ps_upd:
                    log_kinetic_energy += (p**2).sum()
                log_kinetic_energy_prev = 0
                for p in ps_init:
                    log_kinetic_energy_prev += (p**2).sum()
                log_accept_ratio = (logprob - 0.5 * log_kinetic_energy - logprob_prev + 0.5 * log_kinetic_energy_prev).data[0]
                logu = torch.randn(1).log()[0]
                if logu >= log_accept_ratio: # reject
                    proceed = False
                # otherwise keep new
            if not proceed:
                for p,x_prev in zip([p for p in model.parameters() if p.requires_grad], xs_prev):
                    p.data = x_prev
                logprob = logprob_prev
                grads = grads_prev

        return logprob


    for i in range(burn):
        logprob = thinning(thin, epsilon, lmin, lmax)

    tracks = []
    for i in range(num_samples):
        logprob = thinning(thin, epsilon, lmin, lmax)
        tracks.append([logprob.data[0]]+[p.get().data.clone() for p in model.parameters()])

    return list(zip(*tracks))
