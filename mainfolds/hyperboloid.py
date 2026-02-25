"""Hyperboloid manifold."""

import torch

from mainfolds.base import Manifold
from utils.math_utils import arcosh, cosh, sinh


class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class.

    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K

    c = 1 / K is the hyperbolic curvature.
    """

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def sqdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        # clamp distance to avoid nans in Fermi-Dirac decoder
        return torch.clamp(sqdist, max=50.0)
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
    def sqdistmatrix(self, matrix, c, adj):
        """
        向量化计算矩阵任意两行之间的距离

        参数:
            matrix: 形状为 [n_rows, n_cols] 的张量
            c: 曲率参数

        返回:
            distances: 形状为 [n_rows, n_rows] 的张量
        """
        n_rows = matrix.size(0)

        # 使用广播机制计算所有行对之间的距离
        # 增加维度以便进行广播
        x1 = matrix.unsqueeze(1)  # [n_rows, 1, n_cols]
        x2 = matrix.unsqueeze(0)  # [1, n_rows, n_cols]

        # 计算闵可夫斯基内积
        # 对于hyperboloid流形: -x0*y0 + x1*y1 + ... + xd*yd
        prod = self.minkowski_dot(x2, x1)

        K = 1. / c
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[matrix.dtype])
        sqdist = K * arcosh(theta) ** 2
        sqdist = sqdist.squeeze(dim=2)
        # sqdist = torch.log_(sqdist  + 1.1)

        # sqdist = torch.log_(sqdist + 1.1)
        # sqdist = self.min_max_normalize(sqdist + adj,dim=1)
        # sqdist = torch.abs(sqdist)
        # sqdist = self.min_max_normalize(sqdist)
        if adj is None:
            sqdist = torch.tanh(sqdist)
        else:
            sqdist = torch.tanh(sqdist + adj)
        return sqdist
    # def sqdistmatrix(self, x, c):
    #
    #     x1 = x.unsqueeze(1)
    #     x2 = x.unsqueeze(0)
    #     K = 1. / c
    #     sqrt_c = c ** 0.5
    #     # eys = torch.eye(x.shape[0],device=x.device)
    #     prod = self.minkowski_dot(x2, x1)
    #     # theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
    #     sqdist = K * arcosh(-prod / c )
    #     sqdist = sqdist.squeeze(dim=2)
    #     return torch.tanh(sqdist)
    # def sqdistmatrix(self, x, c):
    #     # o = torch.zeros_like(x)
    #     # x = torch.cat([o[:, 0:1], x], dim=1)
    #     # 增加两个维度，使得 x 能够进行广播
    #     # print(x)
    #     x1 = x.unsqueeze(1)
    #     x2 = x.unsqueeze(0)
    #     K = 1. / c
    #     sqrt_c = c ** 0.5
    #     # 计算所有行向量之间的闵可夫斯基内积
    #     prod = self.minkowski_dot(x2, x1,keepdim=False).norm(dim=-1)
    #     print(prod.size(),prod)
    #     # 计算夹角并避免数值不稳定
    #     theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
    #
    #     # 计算距离的平方
    #     sqdist = K * arcosh(theta)** 2
    #
    #     # sqdist = sqdist.squeeze(dim=2)
    #     # 将距离的平方限制在最大值 50.0 以避免数值问题
    #     # sqdist = torch.clamp(sqdist, max=50.0)
    #     # print(sqdist.size(), sqdist)
    #     # sqdist = 2/(1 + torch.exp(sqdist /sqrt_c))
    #     # sqdist = 2 / (1 + torch.exp(sqdist / sqrt_c))
    #     # mask = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)  # 非对角线掩码
    #     # sqdist = torch.where(
    #     #     (sqdist > 0.9999) & mask,  # 检测接近1的非对角线元素
    #     #     torch.zeros_like(sqdist),  # 符合条件的置0
    #     #     sqdist  # 其他保持原值
    #     # )
    #     # sqdist.fill_diagonal_(1.0)  # 自相似度为1
    #
    #     # return
    #     return torch.sigmoid(torch.clamp(sqdist, max=50.0)/sqrt_c)
    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x

    def proj_tan(self, u, x, c):
        K = 1. / c
        d = x.size(1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
        mask = torch.ones_like(u)
        mask[:, 0] = 0
        vals = torch.zeros_like(u)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
        return vals + mask * u

    def proj_tan0(self, u, c):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

    def expmap(self, u, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)

    def logmap(self, x, y, c):
        K = 1. / c
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c)

    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    def mobius_add(self, x, y, c):
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_matvec(self, m, x, c):
        u = self.logmap0(x, c)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu, c)

    def ptransp(self, x, y, u, c):
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def ptransp0(self, x, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm
        v = torch.ones_like(x)
        v[:, 0:1] = - y_norm
        v[:, 1:] = (sqrtK - x0) * y_normalized
        alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtK
        res = u - alpha * v
        return self.proj_tan(res, x, c)

    def to_poincare(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)

