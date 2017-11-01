import numpy
import torch
from torch.autograd import Variable

def hermgauss(n, ttype=torch.FloatTensor):
    x, w = numpy.polynomial.hermite.hermgauss(n)
    x, w = Variable(ttype(x)), Variable(ttype(w))
    return x, w


def mvhermgauss(H, D, ttype=torch.FloatTensor):
    """
    Return the evaluation locations 'xn', and weights 'wn' for a multivariate
    Gauss-Hermite quadrature.

    The outputs can be used to approximate the following type of integral:
    int exp(-x)*f(x) dx ~ sum_i w[i,:]*f(x[i,:])

    :param H: Number of Gauss-Hermite evaluation points.
    :param D: Number of input dimensions. Needs to be known at call-time.
    :return: eval_locations 'x' ((H**D)xD), weights 'w' (H**D)
    """
    gh_x, gh_w = hermgauss(H)
    x = numpy.array(numpy.meshgrid(*(D*(gh_x,))))
    w = numpy.array(numpy.meshgrid(*(D*(gh_w,)))).prod(1)
    x, w = Variable(ttype(x)), Variable(ttype(w))
    return x, w


def mvnquad(f, means, covs, H, Din, Dout=()):
    """
    Computes N Gaussian expectation integrals of a single function 'f'
    using Gauss-Hermite quadrature.
    :param f: integrand function. Takes one input of shape ?xD.
    :param means: NxD
    :param covs: NxDxD
    :param H: Number of Gauss-Hermite evaluation points.
    :param Din: Number of input dimensions. Needs to be known at call-time.
    :param Dout: Number of output dimensions. Defaults to (). Dout is assumed
    to leave out the item index, i.e. f actually maps (?xD)->(?x*Dout).
    :return: quadratures (N,*Dout)
    """
    xn, wn = mvhermgauss(H, Din)
    N = means.size(0)

    # transform points based on Gaussian parameters
    Xt = []
    for c in covs:
        chol_cov = torch.potrf(c, upper=False) # DxD each
        Xt.append(torch.matmul(chol_cov, xn.t()))
    Xt = torch.stack(Xt, dim=0) # NxDx(H**D)
    X = 2.0 ** 0.5 * Xt + means.unsqueeze(2)  # NxDx(H**D)
    Xr = X.permute(2, 0, 1).view(-1, Din)  # (H**D*N)xD

    # perform quadrature
    fX = f(Xr).view(*((H ** Din, N,) + Dout))
    wr = (wn * float(numpi.pi) ** (-Din * 0.5)).view(*((-1,) + (1,) * (1 + len(Dout))))
    return (fX * wr).sum(0)
