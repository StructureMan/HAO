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
        """
        Constructor for the Hyperboloid manifold.

        Initializes attributes specific to the hyperboloid model of hyperbolic geometry.
        The hyperboloid is defined by the equation: -x0^2 + x1^2 + ... + xd^2 = -K,
        where K > 0 is a constant related to curvature. The curvature parameter c is defined as c = 1 / K.

        This class inherits from the base Manifold class and sets up numerical tolerances,
        minimum/maximum norms, and assigns a name for identification.
        """
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'  # Name of the manifold
        # Epsilon values for numerical stability based on data type (float32 or float64)
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15  # Minimum value to avoid division by zero in norm operations
        self.max_norm = 1e6    # Maximum value to clamp large vector norms

    def minkowski_dot(self, x, y, keepdim=True):
        """
        Compute the Minkowski inner product (Lorentzian inner product) between two vectors.

        The Minkowski inner product is defined as:
            ⟨x, y⟩_M = sum(x_i * y_i for i > 0) - x_0 * y_0

        This operation is fundamental in hyperbolic geometry (e.g., on the hyperboloid model),
        where it replaces the standard Euclidean inner product and respects the non-Euclidean metric.

        Args:
            x (torch.Tensor): First input vector.
            y (torch.Tensor): Second input vector.
            keepdim (bool): Whether to retain the last dimension in the output tensor.

        Returns:
            torch.Tensor: The result of the Minkowski inner product.
        """
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        """
        Compute the Minkowski norm (Lorentzian norm) of a vector.

        This norm is derived from the Minkowski inner product and is defined as the square root
        of the inner product of the vector with itself. It respects the hyperbolic geometry
        of the space and is used in operations such as distance computation and normalization.

        Args:
            u (torch.Tensor): Input vector.
            keepdim (bool): Whether to retain the last dimension in the output tensor.

        Returns:
            torch.Tensor: The Minkowski norm of the input vector.
        """
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def sqdist(self, x, y, c):
        """
        Compute the squared geodesic distance between two points on the hyperboloid manifold.

        This distance is derived from the Minkowski inner product and uses the inverse hyperbolic cosine (arcosh)
        to compute a Riemannian distance that respects the curvature of the hyperbolic space.

        Args:
            x (torch.Tensor): First point on the manifold.
            y (torch.Tensor): Second point on the manifold.
            c (float): Curvature of the manifold (c > 0 for hyperbolic space).

        Returns:
            torch.Tensor: Squared geodesic distance between x and y, clamped for numerical stability.
        """
        K = 1. / c
        prod = self.minkowski_dot(x, y)               # ⟨x, y⟩_M
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])  # Ensure valid input to arcosh
        sqdist = K * arcosh(theta) ** 2                # Squared distance
        return torch.clamp(sqdist, max=50.0)           # Clamp large values to avoid NaNs
    def sqdistmatrix(self, x, c):
        """
        Compute the pairwise squared geodesic distances between all points in the input tensor.

        This method computes a full distance matrix where each element (i,j) corresponds to the
        squared geodesic distance between points x[i] and x[j] on the hyperboloid manifold.

        The distance is based on the Minkowski inner product and uses the arcosh function,
        scaled by curvature [c], to respect the non-Euclidean geometry of the space.

        A sigmoid normalization is applied to the final distances to map them into a bounded range [0, 1],
        which can be useful for visualization or similarity-based tasks.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim), representing points on the manifold.
            c (float): Curvature of the hyperboloid manifold (c > 0).

        Returns:
            torch.Tensor: A (batch_size, batch_size) matrix of normalized pairwise distances.
        """
        x1 = x.unsqueeze(1)               # Add dimension for broadcasting (batch, 1, dim)
        x2 = x.unsqueeze(0)               # Add dimension for broadcasting (1, batch, dim)

        K = 1. / c                        # Inverse curvature
        sqrt_c = c ** 0.5                 # Square root of curvature for scaling

        prod = self.minkowski_dot(x2, x1) # Compute Minkowski inner product across all pairs
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])  # Clamp to ensure valid arcosh input
        sqdist = K * arcosh(theta) ** 2   # Compute squared distances
        sqdist = sqdist.squeeze(dim=2)    # Remove extra dimension

        return torch.sigmoid(sqdist / sqrt_c)  # Normalize distances using sigmoid to [0, 1] range

    def proj(self, x, c):
        """
        Project a point onto the hyperboloid manifold.

        This method ensures that a given point `x` lies on the hyperboloid defined by:
            -x0^2 + x1^2 + ... + xd^2 = -K, where K = 1 / c

        The projection fixes the time-like first coordinate `x0` to satisfy the constraint,
        while leaving the remaining space-like coordinates unchanged.

        Args:
            x (torch.Tensor): Input tensor representing points in ambient (Minkowski) space.
            c (float): Curvature of the hyperboloid manifold (c > 0).

        Returns:
            torch.Tensor: A projected point on the hyperboloid manifold.
        """
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)  # Extract spatial components (x1, ..., xd)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2  # Squared norm of spatial components

        mask = torch.ones_like(x)
        mask[:, 0] = 0  # Mask to zero out x0 before adding back corrected value

        vals = torch.zeros_like(x)
        # Compute valid x0 such that the point lies on the hyperboloid
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))

        return vals + mask * x  # Replace original x0 with the computed valid one

    def proj_tan(self, u, x, c):
        """
        Project a vector `u` onto the tangent space at point `x` on the hyperboloid manifold.

        This projection ensures that the resulting vector lies in the tangent space of `x`,
        which satisfies the condition: ⟨x, u⟩_M = 0 using the Minkowski inner product.

        Args:
            u (torch.Tensor): Input vector to be projected.
            x (torch.Tensor): Point on the hyperboloid where the tangent space is defined.
            c (float): Curvature of the hyperboloid manifold (c > 0).

        Returns:
            torch.Tensor: The input vector projected onto the tangent space at [x].
        """
        K = 1. / c
        d = x.size(1) - 1
        # Compute inner product of spatial components: sum(x_i * u_i for i > 0)
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)

        mask = torch.ones_like(u)
        mask[:, 0] = 0  # Mask to zero out original u0 before replacement

        vals = torch.zeros_like(u)
        # Compute valid u0 such that ⟨x, u⟩_M = 0 (Minkowski orthogonality)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])

        return vals + mask * u  # Replace original u0 with computed valid value

    def proj_tan0(self, u, c):
        """
        Project a vector onto the tangent space at the origin (x0 = sqrt(K)) of the hyperboloid manifold.

        At the origin, the tangent space constraint simplifies to setting the time-like component (u0) to zero.
        This method ensures that only the spatial components of `u` remain, effectively projecting it into
        the Euclidean-like tangent space at the origin.

        Args:
            u (torch.Tensor): Input vector to be projected.
            c (float): Curvature of the hyperboloid manifold (c > 0).

        Returns:
            torch.Tensor: The input vector projected onto the tangent space at the origin.
        """
        narrowed = u.narrow(-1, 0, 1)   # Extract the time-like component (u0)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed        # Store original u0 (to be subtracted)
        return u - vals                # Zero out the time-like component

    def expmap(self, u, x, c):
        """
        Compute the exponential map of a tangent vector `u` at point `x` on the hyperboloid manifold.

        The exponential map moves a point along the geodesic defined by the tangent vector `u`,
        generalizing Euclidean vector addition to curved hyperbolic space.

        Args:
            u (torch.Tensor): Tangent vector at point `x`.
            x (torch.Tensor): Point on the hyperboloid manifold.
            c (float): Curvature of the hyperboloid manifold (c > 0).

        Returns:
            torch.Tensor: New point on the manifold obtained by applying the exponential map.
        """
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)                     # Norm of the tangent vector
        normu = torch.clamp(normu, max=self.max_norm)      # Avoid overly large values
        theta = normu / sqrtK                              # Hyperbolic angle
        theta = torch.clamp(theta, min=self.min_norm)      # Prevent division by zero

        # Compute new point on the manifold using hyperbolic trigonometric functions
        result = cosh(theta) * x + sinh(theta) * u / theta

        return self.proj(result, c)  # Project back onto the hyperboloid

    def logmap(self, x, y, c):
        """
        Compute the logarithmic map from point [x] to point `y` on the hyperboloid manifold.

        The logarithmic map returns the tangent vector at `x` that points in the direction of the geodesic
        from `x` to `y`. This operation is the inverse of the exponential map and is used in various
        Riemannian operations such as distance computation and optimization on manifolds.

        Args:
            x (torch.Tensor): Source point on the hyperboloid manifold.
            y (torch.Tensor): Target point on the hyperboloid manifold.
            c (float): Curvature of the hyperboloid manifold (c > 0).

        Returns:
            torch.Tensor: Tangent vector at [x] pointing toward [y] along the geodesic.
        """
        K = 1. / c
        # Adjusted Minkowski inner product to ensure numerical stability
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K

        # Compute the initial vector in ambient space
        u = y + xy * x * c

        # Normalize the vector using the Minkowski norm
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)

        # Compute geodesic distance between x and y
        dist = self.sqdist(x, y, c) ** 0.5

        # Scale the vector to match the geodesic distance
        result = dist * u / normu

        # Project the resulting vector onto the tangent space at x
        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):
        """
        Compute the exponential map from the origin of the hyperboloid manifold.

        This method maps a tangent vector `u` from the tangent space at the origin to
        a point on the hyperboloid along the geodesic defined by `u`.

        Args:
            u (torch.Tensor): Tangent vector at the origin of the hyperboloid.
            c (float): Curvature of the hyperboloid manifold (c > 0).

        Returns:
            torch.Tensor: A point on the hyperboloid manifold resulting from the exponential map.
        """
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1  # Dimension of spatial components

        # Extract spatial part of the tangent vector and reshape
        x = u.narrow(-1, 1, d).view(-1, d)

        # Compute L2 norm of the spatial components
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)  # Avoid division by zero

        theta = x_norm / sqrtK  # Hyperbolic angle proportional to norm

        res = torch.ones_like(u)
        # Compute time-like component using cosh(theta)
        res[:, 0:1] = sqrtK * cosh(theta)
        # Compute spatial components using sinh(theta) scaled by direction
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm

        return self.proj(res, c)  # Project final result onto the hyperboloid

    def logmap0(self, x, c):
        """
        Compute the logarithmic map from the origin to a point `x` on the hyperboloid manifold.

        This method computes the tangent vector at the origin that follows the geodesic to point [x].
        It is the inverse operation to [expmap0], and is useful for mapping points back to the tangent space.

        Args:
            x (torch.Tensor): Point on the hyperboloid manifold.
            c (float): Curvature of the hyperboloid manifold (c > 0).

        Returns:
            torch.Tensor: Tangent vector at the origin pointing toward [x] along the geodesic.
        """
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1  # Dimension of spatial components

        # Extract spatial part of the point and reshape
        y = x.narrow(-1, 1, d).view(-1, d)

        # Compute L2 norm of the spatial components
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)  # Avoid division by zero

        # Compute hyperbolic angle using time-like component
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])

        res = torch.zeros_like(x)
        # Compute spatial components of the tangent vector using arcosh(theta)
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm

        return res  # Tangent vector at origin pointing toward x

    def mobius_add(self, x, y, c):
        """
        Perform Möbius addition of two points on the hyperboloid manifold.

        Möbius addition generalizes vector addition to hyperbolic space. It is defined as:
            x ⊕ y = exp_x(ptransp_x(log_0(y)))

        This operation maps point [y] from the tangent space at the origin to the tangent space at [x],
        then exponentiates the result to obtain a point on the manifold.

        Args:
            x (torch.Tensor): Base point on the hyperboloid.
            y (torch.Tensor): Point to be added (also on the hyperboloid).
            c (float): Curvature of the hyperboloid manifold (c > 0).

        Returns:
            torch.Tensor: Result of Möbius addition of [x] and [y].
        """
        u = self.logmap0(y, c)              # Logarithmic map from origin to y
        v = self.ptransp0(x, u, c)          # Parallel transport the vector to x's tangent space
        return self.expmap(v, x, c)          # Exponentiate to get final point on manifold

    def mobius_matvec(self, m, x, c):
        """
        Perform Möbius matrix-vector multiplication on the hyperboloid manifold.

        This operation generalizes the concept of linear transformation (matrix multiplication)
        from Euclidean space to hyperbolic geometry using the hyperboloid model. It enables operations
        like neural network weight transformations while respecting the curvature and structure
        of the hyperbolic space.

        The process follows these steps:
            1. Map point [x] from the hyperboloid to the tangent space at the origin via [logmap0].
            2. Apply the matrix `m` (e.g., a weight matrix) in this Euclidean-like tangent space.
            3. Map the transformed vector back onto the hyperboloid using [expmap0].

        This ensures that the transformation respects the geometry of the space while enabling
        learning and modeling capabilities similar to standard deep learning layers.

        Args:
            m (torch.Tensor): Matrix to apply in the tangent space. Typically represents learned weights,
                            e.g., in a hyperbolic neural network layer. Shape should be compatible
                            with the dimensionality of the tangent space.
            x (torch.Tensor): A point on the hyperboloid manifold being transformed.
            c (float): Curvature of the hyperboloid manifold (c > 0).

        Returns:
            torch.Tensor: A new point on the hyperboloid manifold resulting from the Möbius matrix-vector
                        multiplication. This output maintains the geometric constraints of the space.
        """
        u = self.logmap0(x, c)              # Map x to the origin's tangent space (linearize)
        mu = u @ m.transpose(-1, -2)        # Apply matrix transformation in flat tangent space
        return self.expmap0(mu, c)          # Map the result back to the manifold

    def ptransp(self, x, y, u, c):
        """
        Perform parallel transport of a tangent vector `u` from the tangent space at `x` to the tangent space at `y`
        on the hyperboloid manifold.

        Parallel transport moves a tangent vector along a geodesic while preserving its direction relative to the manifold.
        This operation is essential for Riemannian optimization and maintaining consistency when working with
        tangent vectors from different base points.

        The implementation follows these steps:
            1. Compute the logarithmic maps between [x] and [y] in both directions: log_x(y) and log_y(x).
            2. Calculate the squared geodesic distance between [x] and [y], clamped for numerical stability.
            3. Compute an alignment coefficient [alpha] using the Minkowski inner product of log_x(y) and `u`.
            4. Adjust `u` based on the curvature-aware correction term involving both log vectors.
            5. Project the resulting vector onto the tangent space at [y].

        Args:
            x (torch.Tensor): Initial base point on the hyperboloid (source point).
            y (torch.Tensor): Target point on the hyperboloid where the vector will be transported.
            u (torch.Tensor): Tangent vector at [x] to be transported.
            c (float): Curvature of the hyperboloid manifold (c > 0).

        Returns:
            torch.Tensor: The input vector `u` parallel-transported from [x] to the tangent space at [y],
                        ensuring it respects the geometry of the manifold.
        """
        logxy = self.logmap(x, y, c)          # Logarithmic map from x to y
        logyx = self.logmap(y, x, c)          # Logarithmic map from y to x
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)  # Clamped squared distance

        alpha = self.minkowski_dot(logxy, u) / sqdist  # Alignment coefficient

        res = u - alpha * (logxy + logyx)     # Adjust u to respect curvature and geodesic path
        return self.proj_tan(res, y, c)       # Project final result into tangent space at y

    def ptransp0(self, x, u, c):
        """
        Perform parallel transport of a tangent vector `u` from the origin to the tangent space at point `x`
        on the hyperboloid manifold.

        This method enables transporting vectors from the origin's tangent space to another point's tangent space,
        preserving their geometric properties relative to the manifold. It is particularly useful in hyperbolic neural networks
        and optimization tasks where maintaining directional consistency across different tangent spaces is important.

        The algorithm computes:
            1. A helper vector `v` that encodes the direction and curvature relationship between the origin and [x].
            2. An alignment coefficient [alpha] based on how much `u` aligns with this direction.
            3. Adjust `u` using [alpha] and `v` to produce a curvature-aware corrected vector.
            4. Project the result into the tangent space at [x].

        Args:
            x (torch.Tensor): Target point on the hyperboloid where the vector will be transported to.
            u (torch.Tensor): Tangent vector at the origin to be transported.
            c (float): Curvature of the hyperboloid manifold (c > 0).

        Returns:
            torch.Tensor: The input vector `u` parallel-transported to the tangent space at [x],
                        ensuring it respects the geometry of the hyperboloid.
        """
        K = 1. / c
        sqrtK = K ** 0.5

        x0 = x.narrow(-1, 0, 1)  # Extract time-like component (x0) of point x

        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)  # Extract spatial components (x1 ... xd)

        # Compute L2 norm of spatial components, clamped for numerical stability
        y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm  # Normalize spatial components

        v = torch.ones_like(x)
        # Set time-like part of correction vector v to negative norm (curvature adjustment)
        v[:, 0:1] = -y_norm
        # Set spatial part of v based on deviation from origin and normalized direction
        v[:, 1:] = (sqrtK - x0) * y_normalized

        # Compute alignment coefficient alpha based on normalized spatial projection
        alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtK

        # Adjust u by subtracting the curvature-aware correction term
        res = u - alpha * v

        # Project final result into the tangent space at x to ensure geometric validity
        return self.proj_tan(res, x, c)

    def to_poincare(self, x, c):
        """
        Map a point from the hyperboloid model to the Poincaré ball model of hyperbolic space.

        This method performs a stereographic projection from the hyperboloid model to the Poincaré ball,
        preserving the intrinsic hyperbolic geometry. The mapping is curvature-dependent and allows
        interoperability between different representations of hyperbolic embeddings.

        Args:
            x (torch.Tensor): A point on the hyperboloid manifold.
            c (float): Curvature of the hyperbolic space (c > 0).

        Returns:
            torch.Tensor: A point in the Poincaré ball model with coordinates in the range (-1, 1),
                        representing the same geometric content as [x] but under a different model.
        """
        K = 1. / c                    # Inverse curvature
        sqrtK = K ** 0.5              # Square root of inverse curvature
        d = x.size(-1) - 1            # Dimensionality of spatial components

        return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)

