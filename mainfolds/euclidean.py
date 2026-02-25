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
        """
        对张量进行 Min-Max 归一化，将其缩放到 [0, 1] 区间

        参数:
            tensor: 输入张量
            dim: 沿指定维度计算最小最大值 (None 表示全局归一化)
            epsilon: 防止除零的小常数

        返回:
            归一化后的张量
        """
        if dim is not None:
            # 沿指定维度计算最小最大值
            min_val = data.min(dim=dim, keepdim=True).values
            max_val = data.max(dim=dim, keepdim=True).values
        else:
            # 全局最小最大值
            min_val = data.min()
            max_val = data.max()

        # 防止除零
        range_val = max_val - min_val
        range_val = torch.where(range_val == 0, torch.ones_like(range_val) * epsilon, range_val)

        # 应用 Min-Max 公式
        normalized = (data - min_val) / range_val

        return normalized

    def mean_normalize(self, data, dim=None, epsilon=1e-8):
        """
        对张量进行 Mean-Normalization，将其缩放到 [-1, 1] 区间附近，均值为 0

        参数:
            tensor: 输入张量
            dim: 沿指定维度计算均值、最小最大值 (None 表示全局归一化)
            epsilon: 防止除零的小常数

        返回:
            归一化后的张量
        """
        if dim is not None:
            # 沿指定维度计算均值、最小最大值
            mean_val = data.mean(dim=dim, keepdim=True)
            min_val = data.min(dim=dim, keepdim=True).values
            max_val = data.max(dim=dim, keepdim=True).values
        else:
            # 全局均值、最小最大值
            mean_val = data.mean()
            min_val = data.min()
            max_val = data.max()

        # 防止除零
        range_val = max_val - min_val
        range_val = torch.where(range_val == 0, torch.ones_like(range_val) * epsilon, range_val)

        # 应用 Mean-Normalization 公式
        normalized = (data - mean_val) / range_val

        return normalized

    def sym_norm_adj(self,adj, add_self_loop=True):
        """
        adj: torch.FloatTensor (n,n)  0/1 或权重，对称
        return 对称归一化邻接矩阵 Â
        """
        if add_self_loop:
            adj = adj + torch.eye(adj.size(0), device=adj.device)
        deg = adj.sum(1)  # 度向量
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return D_inv_sqrt @ adj @ D_inv_sqrt

    def sqdistmatrix(self, p, c,adj):
        # 计算矩阵p中任意两行之间的平方距离
        # p的维度为 (numvectors, numfeatures)
        # 使用广播机制计算所有行对之间的差的平方
        sqrt_c = c ** 0.5
        sqdist = (p.unsqueeze(1) - p.unsqueeze(0)).pow(2).sum(-1)
        # sqdist = torch.log_(sqdist + 1.1)
        # sqdist = 2 / (1 + torch.exp(sqdist / sqrt_c))
        # mask = ~torch.eye(p.size(0), dtype=torch.bool, device=p.device)  # 非对角线掩码
        # sqdist = torch.where(
        #     (sqdist > 0.9999) & mask,  # 检测接近1的非对角线元素
        #     torch.zeros_like(sqdist),  # 符合条件的置0
        #     sqdist  # 其他保持原值
        # )
        # sqdist.fill_diagonal_(1.0)
        # sqdist = torch.tanh(sqdist + adj)
        # sqdist = sqdist + adj
        # sqdist = self.min_max_normalize(sqdist + adj,dim=0)
        # sqdist = self.min_max_normalize(sqdist)
        if adj is None:
            sqdist = torch.tanh(sqdist)
        else:
            sqdist = torch.tanh(sqdist + adj)
        return sqdist

    def z_score_normalize(self,data, dim=None, epsilon=1e-8):
        """
        对张量进行 Z-Score 归一化（均方差归一化）
        将数据转换为均值为 0，标准差为 1 的分布

        参数:
            tensor: 输入张量
            dim: 沿指定维度计算均值和标准差 (None 表示全局归一化)
            epsilon: 防止除零的小常数

        返回:
            归一化后的张量
        """
        if dim is not None:
            # 沿指定维度计算均值和标准差
            mean = data.mean(dim=dim, keepdim=True)
            std = data.std(dim=dim, keepdim=True, unbiased=False)
        else:
            # 全局均值和标准差
            mean = data.mean()
            std = data.std(unbiased=False)

        # 防止除零
        std = torch.where(std == 0, torch.ones_like(std) * epsilon, std)

        # 应用 Z-Score 公式
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
