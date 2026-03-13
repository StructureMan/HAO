"""Euclidean manifold."""
import torch

from mainfolds.base import Manifold

class Euclidean(Manifold):
    """
    Euclidean Manifold class.
    """

    def __init__(self):
        super(Euclidean, self).__init__()
        self.name = 'Euclidean'

    def normalize(self, p):
        dim = p.size(-1)
        p.view(-1, dim).renorm_(2, 0, 1.)
        return p

    def sqdist(self, p1, p2, c):
        return (p1 - p2).pow(2).sum(dim=-1)

    def egrad2rgrad(self, p, dp, c):
        return dp

    def min_max_normalize(self, data, dim=None, epsilon=1e-8):

        if dim is not None:

            min_val = data.min(dim=dim, keepdim=True).values
            max_val = data.max(dim=dim, keepdim=True).values
        else:

            min_val = data.min()
            max_val = data.max()


        range_val = max_val - min_val
        range_val = torch.where(range_val == 0, torch.ones_like(range_val) * epsilon, range_val)


        normalized = (data - min_val) / range_val

        return normalized

    def mean_normalize(self, data, dim=None, epsilon=1e-8):

        if dim is not None:

            mean_val = data.mean(dim=dim, keepdim=True)
            min_val = data.min(dim=dim, keepdim=True).values
            max_val = data.max(dim=dim, keepdim=True).values
        else:

            mean_val = data.mean()
            min_val = data.min()
            max_val = data.max()


        range_val = max_val - min_val
        range_val = torch.where(range_val == 0, torch.ones_like(range_val) * epsilon, range_val)


        normalized = (data - mean_val) / range_val

        return normalized

    def sym_norm_adj(self,adj, add_self_loop=True):

        if add_self_loop:
            adj = adj + torch.eye(adj.size(0), device=adj.device)
        deg = adj.sum(1)  
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return D_inv_sqrt @ adj @ D_inv_sqrt

    def sqdistmatrix(self, p, c,adj):

        sqrt_c = c ** 0.5
        sqdist = (p.unsqueeze(1) - p.unsqueeze(0)).pow(2).sum(-1)
        if adj is None:
            sqdist = torch.tanh(sqdist)
        else:
            sqdist = torch.tanh(sqdist + adj)
        return sqdist

    def z_score_normalize(self,data, dim=None, epsilon=1e-8):

        if dim is not None:

            mean = data.mean(dim=dim, keepdim=True)
            std = data.std(dim=dim, keepdim=True, unbiased=False)
        else:

            mean = data.mean()
            std = data.std(unbiased=False)


        std = torch.where(std == 0, torch.ones_like(std) * epsilon, std)


        normalized = (data - mean) / std

        return normalized
    def proj(self, p, c):
        return p

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        return p + u

    def logmap(self, p1, p2, c):
        return p2 - p1

    def expmap0(self, u, c):
        return u

    def logmap0(self, p, c):
        return p

    def mobius_add(self, x, y, c, dim=-1):
        return x + y

    def mobius_matvec(self, m, x, c):
        mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def inner(self, p, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, v, c):
        return v

    def ptransp0(self, x, v, c):
        return x + v
