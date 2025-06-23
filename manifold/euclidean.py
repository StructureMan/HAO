"""Euclidean manifold."""
import torch

from mainfolds.base import Manifold

class Euclidean(Manifold):
    """
    Euclidean Manifold class.
    """

    def __init__(self):
        """
        Constructor for the Euclidean manifold class.

        Initializes the Euclidean manifold with a name identifier.
        This class inherits from the base Manifold class and implements
        operations specific to Euclidean geometry.
        """
        super(Euclidean, self).__init__()
        self.name = 'Euclidean'  # Set the name of the manifold to "Euclidean"

    def normalize(self, p):
        """
        Normalize the input tensor so that it lies on the surface of a unit sphere in Euclidean space.

        This method applies L2 normalization along the last dimension of the tensor,
        ensuring that each vector has unit length (norm of 1).

        Args:
            p (torch.Tensor): Input tensor of arbitrary shape, where the last dimension represents feature vectors.

        Returns:
            torch.Tensor: The normalized tensor with unit-length vectors along the last dimension.
        """
        dim = p.size(-1)
        p.view(-1, dim).renorm_(2, 0, 1.)  # Renormalize each row to have L2 norm of 1
        return p

    def sqdist(self, p1, p2, c):
        """
        Compute the squared Euclidean distance between two tensors p1 and p2.

        This function calculates the element-wise squared L2 distance (sum of squared differences)
        along the last dimension, which is typically used to measure similarity between vectors.

        Although the curvature [c] is passed as an argument, it has no effect in standard Euclidean distance.

        Args:
            p1 (torch.Tensor): First input tensor of shape (..., dim), representing points in Euclidean space.
            p2 (torch.Tensor): Second input tensor of shape (..., dim), representing points in Euclidean space.
            c (float): Curvature parameter (not used in Euclidean space).

        Returns:
            torch.Tensor: Tensor containing the squared Euclidean distances between corresponding points in p1 and p2.
        """
        return (p1 - p2).pow(2).sum(dim=-1)  # Squared Euclidean distance

    def egrad2rgrad(self, p, dp, c):
        """
        Convert Euclidean gradient to Riemannian gradient.

        In Euclidean space, the Euclidean gradient (egrad) and Riemannian gradient (rgrad) are the same,
        since the space is flat and does not require any geometric correction.

        Args:
            p (torch.Tensor): Point on the manifold (not used in Euclidean space).
            dp (torch.Tensor): Euclidean gradient (tangent vector at point p).
            c (float): Curvature parameter (ignored in Euclidean space).

        Returns:
            torch.Tensor: The Riemannian gradient, which is identical to the input Euclidean gradient.
        """
        return dp  # No transformation needed in Euclidean space

    def sqdistmatrix(self, p, c):
        """
        Compute the pairwise squared Euclidean distance matrix between rows of p.

        This function calculates the squared Euclidean distance between every pair of rows
       

        Args:
            p (torch.Tensor): Input tensor of shape (num_vectors, num_features), representing points in Euclidean space.
            c (float): Curvature parameter used to scale distances. Has no strict geometric meaning in Euclidean space,
                    but may be used for scaling or normalization purposes in certain applications.

        Returns:
            torch.Tensor: A square matrix of shape (num_vectors, num_vectors) containing
                        the scaled pairwise squared distances between vectors, passed through a sigmoid function.
        """
        sqrt_c = c ** 0.5
        # Compute pairwise squared Euclidean distances using broadcasting
        sqdist = (p.unsqueeze(1) - p.unsqueeze(0)).pow(2).sum(-1)
        # Scale distances by curvature and apply sigmoid to bound values between 0 and 1
        return torch.sigmoid(sqdist / sqrt_c)
    def proj(self, p, c):
        """
        Project a point onto the Euclidean manifold.

        In Euclidean space, projection has no effect because all points are already valid in this geometry.
        This method is included for interface consistency with other manifolds that require projection.

        Args:
            p (torch.Tensor): Input tensor representing points in space.
            c (float): Curvature parameter (not applicable in Euclidean space).

        Returns:
            torch.Tensor: The input tensor `p`, unchanged, as Euclidean space requires no projection.
        """
        return p  # No projection needed in Euclidean space

    def proj_tan(self, u, p, c):
        """
        Project a vector onto the tangent space at point p in Euclidean space.

        In Euclidean geometry, all vectors are already in the tangent space and require no projection.
        This method is provided for interface consistency with other manifolds that do require tangent projection.

        Args:
            u (torch.Tensor): Input vector to be projected (already valid in Euclidean tangent space).
            p (torch.Tensor): Point on the manifold (not used in Euclidean space).
            c (float): Curvature parameter (not applicable in Euclidean space).

        Returns:
            torch.Tensor: The input vector `u`, unchanged, as no projection is needed in Euclidean space.
        """
        return u  # No tangent projection required in Euclidean space

    def proj_tan0(self, u, c):
        """
        Project a vector onto the tangent space at the origin in Euclidean space.

        In Euclidean geometry, the tangent space at any point (including the origin) is the same as the original space,
        so no actual projection is needed. This method is provided for interface consistency with other manifolds.

        Args:
            u (torch.Tensor): Input vector to be projected (already valid in Euclidean space).
            c (float): Curvature parameter (not applicable in Euclidean space).

        Returns:
            torch.Tensor: The input vector `u`, unchanged, as no projection is required in Euclidean space.
        """
        return u  # No projection needed in Euclidean space

    def expmap(self, u, p, c):
        """
        Compute the exponential map in Euclidean space.

        In Euclidean geometry, the exponential map from a point `p` in direction `u` simplifies to vector addition.
        This method moves along the straight-line geodesic starting at `p` in the direction of `u`.

        Args:
            u (torch.Tensor): Tangent vector at point p.
            p (torch.Tensor): Point on the manifold (starting point for the exponential map).
            c (float): Curvature parameter (not applicable in Euclidean space).

        Returns:
            torch.Tensor: The resulting point after applying the exponential map, equivalent to `p + u`.
        """
        return p + u  # In Euclidean space, expmap is simply vector addition

    def logmap(self, p1, p2, c):
        """
        Compute the logarithmic map from point p1 to point p2 in Euclidean space.

        The logarithmic map in Euclidean geometry is simply the vector pointing from `p1` to `p2`,
        which represents the shortest path (geodesic) between the two points.

        Args:
            p1 (torch.Tensor): Starting point on the manifold.
            p2 (torch.Tensor): End point on the manifold.
            c (float): Curvature parameter (not applicable in Euclidean space).

        Returns:
            torch.Tensor: The tangent vector representing the geodesic from `p1` to `p2`, equivalent to `p2 - p1`.
        """
        return p2 - p1  # In Euclidean space, logmap is simply vector subtraction

    def logmap(self, p1, p2, c):
        """
        Compute the logarithmic map from point p1 to point p2 in Euclidean space.

        The logarithmic map in Euclidean geometry is simply the vector pointing from `p1` to `p2`,
        which represents the shortest path (geodesic) between the two points.

        Args:
            p1 (torch.Tensor): Starting point on the manifold.
            p2 (torch.Tensor): End point on the manifold.
            c (float): Curvature parameter (not applicable in Euclidean space).

        Returns:
            torch.Tensor: The tangent vector representing the geodesic from `p1` to `p2`, equivalent to `p2 - p1`.
        """
        return p2 - p1  # In Euclidean space, logmap is simply vector subtraction

    def logmap0(self, p, c):
        """
        Compute the logarithmic map from the origin to a point p in Euclidean space.

        In Euclidean geometry, the logarithmic map from the origin to point `p` is simply the vector `p` itself,
        since the origin is at zero and the geodesic between them is just the straight line represented by `p`.

        Args:
            p (torch.Tensor): Point on the manifold (vector from the origin).
            c (float): Curvature parameter (not applicable in Euclidean space).

        Returns:
            torch.Tensor: The tangent vector from the origin to point `p`, which is the same as the input `p`.
        """
        return p  # In Euclidean space, logmap0(p) = p

    def mobius_add(self, x, y, c, dim=-1):
        """
        Perform Möbius addition in Euclidean space.

        In Euclidean geometry, Möbius addition simplifies to standard vector addition,
        as there is no curvature to account for in flat space.

        Args:
            x (torch.Tensor): First input tensor (point or vector).
            y (torch.Tensor): Second input tensor (point or vector).
            c (float): Curvature parameter (not applicable in Euclidean space).
            dim (int): Dimension along which the operation is applied (not used in Euclidean space).

        Returns:
            torch.Tensor: Result of Möbius addition, equivalent to `x + y` in Euclidean space.
        """
        return x + y  # No special behavior needed; Euclidean space uses standard vector addition

    def mobius_matvec(self, m, x, c):
        """
        Perform Möbius matrix-vector multiplication in Euclidean space.

        In Euclidean geometry, Möbius operations reduce to standard linear algebra operations.
        This method applies a regular matrix-to-vector multiplication.

        Args:
            m (torch.Tensor): Matrix to be multiplied with the vector.
            x (torch.Tensor): Input vector or batch of vectors.
            c (float): Curvature parameter (not applicable in Euclidean space).

        Returns:
            torch.Tensor: Result of matrix-vector multiplication, equivalent to `x @ m.T` in Euclidean space.
        """
        mx = x @ m.transpose(-1, -2)  # Standard matrix-vector multiplication
        return mx

    def init_weights(self, w, c, irange=1e-5):
        """
        Initialize the weights of a tensor with small uniform values.

        In Euclidean space, weight initialization follows standard practices and involves
        sampling from a uniform distribution within a small range.

        Args:
            w (torch.Tensor): Weight tensor to be initialized.
            c (float): Curvature parameter (not applicable in Euclidean space).
            irange (float): Range of the uniform distribution used for initialization.

        Returns:
            torch.Tensor: The input weight tensor `w` with initialized values.
        """
        w.data.uniform_(-irange, irange)  # Initialize with small random values
        return w

    def inner(self, p, c, u, v=None, keepdim=False):
        """
        Compute the inner product (dot product) between two vectors in Euclidean space.

        This method calculates the standard Euclidean inner product between vectors `u` and `v`.
        If `v` is not provided, it defaults to using `u`, effectively computing the squared norm.

        Args:
            p (torch.Tensor): Point on the manifold (not used in Euclidean space).
            c (float): Curvature parameter (not applicable in Euclidean space).
            u (torch.Tensor): First input vector.
            v (torch.Tensor, optional): Second input vector. Defaults to `u` if not provided.
            keepdim (bool, optional): Whether to retain the last dimension in the output. Defaults to False.

        Returns:
            torch.Tensor: The inner product of `u` and `v`, computed as their element-wise product summed along the last dimension.
        """
        if v is None:
            v = u  # Compute squared norm if v is not provided
        return (u * v).sum(dim=-1, keepdim=keepdim)  # Standard Euclidean inner product

    def ptransp(self, x, y, v, c):
        """
        Perform parallel transport of a vector in Euclidean space.

        In Euclidean geometry, parallel transport has no effect because the space is flat.
        This method simply returns the input vector `v` unchanged.

        Args:
            x (torch.Tensor): Start point on the manifold (not used in Euclidean space).
            y (torch.Tensor): End point on the manifold (not used in Euclidean space).
            v (torch.Tensor): Vector to be transported.
            c (float): Curvature parameter (not applicable in Euclidean space).

        Returns:
            torch.Tensor: The transported vector, which is the same as the input vector `v`.
        """
        return v  # No transformation needed in Euclidean space

    def ptransp0(self, x, v, c):
        """
        Perform parallel transport from the origin to a point `x` in Euclidean space.

        In Euclidean geometry, parallel transport does not alter vectors since the space is flat.


        Args:
            x (torch.Tensor): Target point in Euclidean space where the vector is transported.
            v (torch.Tensor): Vector to be transported.
            c (float): Curvature parameter (not applicable in Euclidean space).

        Returns:
            torch.Tensor: The transported vector at point [x], equivalent to `x + v` in Euclidean space.
        """
        return x + v  # In Euclidean space, transporting a vector from origin to x is just vector addition
