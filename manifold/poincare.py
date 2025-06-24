"""Poincare ball manifold."""

import torch

from mainfolds.base import Manifold
from utils.math_utils import artanh, tanh


class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self):
        """
        Constructor for the Poincaré Ball manifold.

        Initializes properties specific to the Poincaré ball model of hyperbolic geometry.
        The Poincaré ball is defined as the set of points satisfying:
            x0^2 + x1^2 + ... + xd^2 < 1 / c

        where [c] is the curvature (c > 0), and the ball has radius `1/sqrt(c)`.

        This class inherits from the base Manifold class and sets up numerical tolerances,
        minimum norm thresholds, and assigns a name for identification.

        Attributes:
            name (str): Identifier for the manifold, set to 'PoincareBall'.
            min_norm (float): Minimum value used to clamp vector norms in operations like logmap/expmap
                            to avoid division by zero.
            eps (dict): Dictionary specifying small epsilon values based on tensor data type
                        (float32 or float64) for numerical stability during computations.
        """
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'          # Name of the manifold
        self.min_norm = 1e-15               # Minimum threshold to prevent division by zero
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}  # Epsilon values per data type for stability

    def sqdist(self, p1, p2, c):
        """
        Compute the squared geodesic distance between two points on the Poincaré ball manifold.

        This method calculates the intrinsic hyperbolic distance that respects the curvature [c],
        using the analytic formula derived from the Poincaré ball model. The distance is based on
        Möbius addition and the inverse hyperbolic tangent (artanh) function.

        The computation follows these steps:
            1. Compute the Möbius addition of -p1 and p2, which gives a vector pointing from p1 to p2.
            2. Take the L2 norm of this vector to measure its magnitude.
            3. Apply the artanh function scaled by sqrt(c) to map it into hyperbolic space.
            4. Scale the result to get the final Riemannian distance and return its square.

        Args:
            p1 (torch.Tensor): First point in the Poincaré ball, shape (..., d).
            p2 (torch.Tensor): Second point in the Poincaré ball, shape (..., d).
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).

        Returns:
            torch.Tensor: Squared geodesic distance between p1 and p2 in the Poincaré ball,
                        shape (...) — same as input, with the last dimension removed after taking the norm.
        """
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        return dist ** 2


    def sqdistmatrix(self, p, c):
        """
        Compute the pairwise squared geodesic distances between all points in the input tensor on the Poincaré ball manifold.

        This method calculates a full distance matrix where each element (i,j) corresponds to the squared geodesic distance
        between points `p[i]` and `p[j]` in the Poincaré ball model. The computation respects the curvature [c]
        and uses Möbius addition as the foundation for hyperbolic vector operations.

        Steps:
            1. Broadcast input tensor `p` into two shapes: (n, 1, d) and (1, n, d) for broadcasting.
            2. Compute the Möbius addition of `-p1` and `p2` across all point pairs to get displacement vectors.
            3. Take the L2 norm of these vectors to measure their magnitudes.
            4. Apply the inverse hyperbolic tangent function scaled by sqrt(c) to map into hyperbolic space.
            5. Scale the result by `2 / sqrt(c)` to obtain the Riemannian distance and return its scaled version.

        Args:
            p (torch.Tensor): Input tensor of shape (n, d), representing [n] points in the Poincaré ball.
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).

        Returns:
            torch.Tensor: A (n, n) matrix of normalized pairwise squared geodesic distances,
                        transformed through sigmoid to map into a bounded range [0, 1].
        """
        sqrt_c = c ** 0.5
        p1 = p.unsqueeze(1)                   # Reshape to (n, 1, d) for broadcasting
        p2 = p.unsqueeze(0)                   # Reshape to (1, n, d) for broadcasting

        midpoints = self.mobius_add(-p1, p2, c)  # Compute Möbius difference vector from p1 to p2

        norm_midpoints = midpoints.norm(dim=-1)    # Compute Euclidean norm over last dimension

        dist_c = artanh(sqrt_c * norm_midpoints)    # Map to hyperbolic space using artanh
        sqdist = dist_c * 2 / sqrt_c               # Scale to get final distance

        return torch.sigmoid(sqdist / sqrt_c)        # Normalize with sigmoid to bound output in [0, 1] 
    def _lambda_x(self, x, c):
        """
        Compute the conformal factor λ(x) for a point `x` on the Poincaré ball manifold.

        The conformal factor λ(x) is a scalar quantity used in various geometric operations such as:
            - Riemannian metric tensor scaling
            - Conversion between Euclidean and Riemannian gradients
            - Exponential and logarithmic map computations

        This factor depends only on the norm of the point [x] and the curvature [c], and ensures numerical stability
        by clamping the denominator to avoid division by zero or near-zero values.

        Args:
            x (torch.Tensor): Input tensor representing points on the Poincaré ball, shape (..., d).
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).

        Returns:
            torch.Tensor: Conformal factor λ(x) computed for each input point, shape (..., 1).
        """
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)  # Squared L2 norm of x
        return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)     # Compute λ(x) with numerical safeguard

    def egrad2rgrad(self, p, dp, c):
        """
        Convert a Euclidean gradient to a Riemannian gradient on the Poincaré ball manifold.

        This method applies the Riemannian metric tensor to transform a Euclidean gradient `dp`
        at point [p] into a Riemannian gradient that respects the geometry of the Poincaré ball model.
        
        The transformation uses the conformal factor λ(x), which depends on the curvature [c]
        and the norm of the point [p], scaling the gradient by `1 / λ(p)^2`.

        Args:
            p (torch.Tensor): Point on the Poincaré ball where the gradient is computed, shape (..., d).
            dp (torch.Tensor): Euclidean gradient at point [p], same shape as [p].
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).

        Returns:
            torch.Tensor: The input gradient converted to a Riemannian gradient, same shape as `dp`.
        """
        lambda_p = self._lambda_x(p, c)         # Compute conformal factor λ(p)
        dp /= lambda_p.pow(2)                   # Scale Euclidean gradient by 1 / λ(p)^2
        return dp                                 # Return Riemannian gradient

    def proj(self, x, c):
        """
        Project a point onto the Poincaré ball manifold.

        This method ensures that the input point `x` lies within the Poincaré ball by clamping its norm
        to be strictly less than the ball radius `1 / sqrt(c)`. Points outside the ball are radially
        projected onto its boundary, while valid points remain unchanged.

        Args:
            x (torch.Tensor): Input tensor representing points in Euclidean space, shape (..., d).
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).

        Returns:
            torch.Tensor: A point projected onto the Poincaré ball, same shape as input.
        """
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)  # Compute L2 norm with numerical safeguard
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)                            # Define maximum allowed norm
        cond = norm > maxnorm                                                    # Identify points outside the ball
        projected = x / norm * maxnorm                                           # Radial projection for out-of-bound points
        return torch.where(cond, projected, x)                                   # Return projected or original points

    def proj_tan(self, u, p, c):
        """
        Identity projection of a tangent vector onto the tangent space at point `p` in the Poincaré ball model.

        In the Poincaré ball manifold, all vectors are considered to lie in the tangent space at any point,
        due to the flat (Euclidean-like) nature of the tangent spaces. Hence, no actual projection is needed,
        and the input vector `u` is returned as-is.

        Args:
            u (torch.Tensor): Input vector to be projected (already in the tangent space).
            p (torch.Tensor): Point on the Poincaré ball where the tangent space is defined.
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).

        Returns:
            torch.Tensor: The original vector `u`, unchanged, as it already lies in the tangent space.
        """
        return u

    def proj_tan0(self, u, c):
        """
        Identity projection of a vector onto the tangent space at the origin of the Poincaré ball.

        In the Poincaré ball model, the tangent space at the origin is equivalent to the ambient Euclidean space.
        Therefore, any input vector `u` is already a valid tangent vector at the origin, and no actual projection is needed.

        Args:
            u (torch.Tensor): Input vector to be projected.
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).

        Returns:
            torch.Tensor: The original vector `u`, unchanged, as it already lies in the tangent space at the origin.
        """
        return u

    def expmap(self, u, p, c):
        """
        Compute the exponential map from a tangent vector `u` at point `p` on the Poincaré ball manifold.

        The exponential map moves a point along the geodesic defined by the tangent vector `u`,
        generalizing Euclidean vector addition to curved hyperbolic space. This operation uses:
            - The conformal factor λ(p) to scale inner products appropriately.
            - Hyperbolic trigonometric functions (tanh) to respect curvature.
            - Möbius addition to perform the final step onto the manifold.

        Args:
            u (torch.Tensor): Tangent vector at point `p`.
            p (torch.Tensor): Point on the Poincaré ball where the tangent vector is based.
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).

        Returns:
            torch.Tensor: A new point on the Poincaré ball obtained by applying the exponential map.
        """
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)  # Compute norm of tangent vector

        # Compute second term using tanh and the conformal factor λ(p)
        second_term = (
            tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
            * u
            / (sqrt_c * u_norm)
        )

        # Apply Möbius addition to move point `p` along the geodesic defined by `u`
        gamma_1 = self.mobius_add(p, second_term, c)

        return gamma_1

    def logmap(self, p1, p2, c):
        """
        Compute the logarithmic map from point `p1` to point `p2` on the Poincaré ball manifold.

        The logarithmic map returns the tangent vector at `p1` that points in the direction of the geodesic
        from `p1` to `p2`. This operation is the inverse of the exponential map and is essential for:
            - Computing distances in hyperbolic space.
            - Riemannian optimization.
            - Parallel transport and interpolation between points.

        The computation involves:
            1. Möbius addition of `-p1` and `p2` to get the displacement vector.
            2. Normalizing this vector and scaling it using curvature-aware terms involving artanh, λ(p1), and sqrt(c).

        Args:
            p1 (torch.Tensor): Source point on the Poincaré ball, shape (..., d).
            p2 (torch.Tensor): Target point on the Poincaré ball, same shape as p1.
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).

        Returns:
            torch.Tensor: Tangent vector at `p1` pointing toward `p2`, same shape as input points.
        """
        sub = self.mobius_add(-p1, p2, c)                         # Displacement vector from p1 to p2
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)  # Norm with numerical safeguard

        lam = self._lambda_x(p1, c)                                # Conformal factor at p1
        sqrt_c = c ** 0.5                                          # Square root of curvature

        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        """
        Compute the exponential map from a tangent vector `u` at the origin of the Poincaré ball.

        This method maps a vector from the tangent space at the origin to the manifold via the exponential map.
        It uses hyperbolic geometry-aware scaling based on curvature [c], ensuring that the resulting point lies within the ball.

        Args:
            u (torch.Tensor): Tangent vector at the origin in Euclidean space.
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).

        Returns:
            torch.Tensor: A point on the Poincaré ball obtained by applying the exponential map at the origin.
        """
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)  # Avoid division by zero
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)  # Map vector to Poincaré ball
        return gamma_1

    def logmap0(self, p, c):
        """
        Compute the logarithmic map from the origin to a point `p` on the Poincaré ball manifold.

        This method returns the tangent vector at the origin that points in the direction of the geodesic
        from the origin to point [p]. It is the inverse of the exponential map at the origin and respects the curvature [c].

        The result is computed by:
            1. Computing the norm of point [p] with numerical safeguard.
            2. Scaling the point using artanh and curvature-aware terms to obtain the corresponding tangent vector.

        Args:
            p (torch.Tensor): Point on the Poincaré ball, shape (..., d).
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).

        Returns:
            torch.Tensor: Tangent vector at the origin pointing toward point [p], same shape as input.
        """
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)  # Norm with numerical safeguard
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm               # Compute scaling factor
        return scale * p                                                     # Return scaled vector

    def mobius_add(self, x, y, c, dim=-1):
        """
        Compute the Möbius addition of two points `x` and `y` in the Poincaré ball model.

        Möbius addition is the analog of vector addition in hyperbolic space and is used to define
        geodesics, exponential/logarithmic maps, and other geometric operations on the manifold.

        The formula follows:
            x ⊕ y = ( (1 + 2c⟨x, y⟩ + c||y||²) * x + (1 - c||x||²) * y )
                    / (1 + 2c⟨x, y⟩ + c²||x||²||y||²)

        where ⟨x, y⟩ denotes the inner product and ||·||² denotes the squared L2 norm.

        Args:
            x (torch.Tensor): First point in the Poincaré ball, shape (..., d).
            y (torch.Tensor): Second point in the Poincaré ball, same shape as [x].
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).
            dim (int): Dimension along which to compute the norm and dot product. Default: -1.

        Returns:
            torch.Tensor: Result of Möbius addition of [x]and [y], same shape as input.
        """
        x2 = x.pow(2).sum(dim=dim, keepdim=True)                 # ||x||²
        y2 = y.pow(2).sum(dim=dim, keepdim=True)                 # ||y||²
        xy = (x * y).sum(dim=dim, keepdim=True)                  # ⟨x, y⟩

        # Numerator: (1 + 2c⟨x, y⟩ + c||y||²)x + (1 - c||x||²)y
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y

        # Denominator: 1 + 2c⟨x, y⟩ + c²||x||²||y||²
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2

        return num / denom.clamp_min(self.min_norm)              # Safe division with clamping

    def mobius_matvec(self, m, x, c):
        """
        Perform Möbius matrix-vector multiplication in the Poincaré ball manifold.

        This operation generalizes the standard matrix-vector multiplication to hyperbolic space.
        It transforms a Euclidean vector [x] under a linear map `m`, then maps the result back onto the Poincaré ball
        using curvature-aware scaling and hyperbolic trigonometric functions.

        The method:
            1. Applies the matrix `m` to the vector [x] (in Euclidean space).
            2. Computes norms before and after transformation.
            3. Scales the transformed vector using tanh and artanh to ensure it lies within the ball.
            4. Handles zero vectors separately for numerical stability.

        Args:
            m (torch.Tensor): Matrix to apply, shape (d, d) or batched matrices (..., d, d).
            x (torch.Tensor): Vector in the Poincaré ball, shape (..., d).
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).

        Returns:
            torch.Tensor: Resulting point on the Poincaré ball after applying the Möbius matrix-vector product.
        """
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)  # Norm of input vector

        mx = x @ m.transpose(-1, -2)                      # Apply matrix multiplication
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)  # Norm of transformed vector

        # Scale transformed vector using hyperbolic-aware tanh/artanh functions
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)

        # Handle zero-vectors by replacing with zeros
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)

        return res

    def init_weights(self, w, c, irange=1e-5):
        """
        Initialize weights uniformly within a small range and return them as-is.

        This method is used to initialize embedding or parameter tensors for models operating in the Poincaré ball.
        It ensures that initial values are close to zero, which helps avoid numerical instability during the first steps of training.

        Although it doesn't explicitly project points into the Poincaré ball, initializing with small values ensures that
        the points are naturally within the ball since its radius is `1 / sqrt(c)` and typically much larger than `irange`.

        Args:
            w (torch.Tensor): Weight tensor to be initialized, shape (..., d).
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).
            irange (float): Range for uniform initialization, i.e., values will be sampled from [-irange, irange].

        Returns:
            torch.Tensor: The initialized weight tensor, unchanged in shape and device.
        """
        w.data.uniform_(-irange, irange)  # Initialize with small random values
        return w

    def _gyration(self, u, v, w, c, dim: int = -1):
        """
        Compute the gyration of vectors `u`, `v`, and `w` in the Poincaré ball manifold.

        Gyration is a three-point invariant in hyperbolic geometry that captures the non-associativity
        of Möbius addition. It is used in operations like parallel transport and vector translation
        on curved manifolds.

        This method computes:
            gyr[u, v]w = w + 2 * (a * u + b * v) / d

        where [a], `b`, and [d] are curvature-dependent terms derived from inner products and norms of the input vectors.

        Args:
            u (torch.Tensor): First vector involved in the gyration.
            v (torch.Tensor): Second vector involved in the gyration.
            w (torch.Tensor): Vector to be gyrated (transformed by the gyration).
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).
            dim (int): Dimension along which vector operations are performed (default: -1).

        Returns:
            torch.Tensor: The result of applying gyration to vectors [u], [v], and [w],
                        same shape as input tensors.
        """
        u2 = u.pow(2).sum(dim=dim, keepdim=True)                   # ||u||²
        v2 = v.pow(2).sum(dim=dim, keepdim=True)                   # ||v||²
        uv = (u * v).sum(dim=dim, keepdim=True)                    # ⟨u, v⟩
        uw = (u * w).sum(dim=dim, keepdim=True)                    # ⟨u, w⟩
        vw = (v * w).sum(dim=dim, keepdim=True)                    # ⟨v, w⟩
        c2 = c ** 2                                                # c²

        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw              # Term 'a' in gyration formula
        b = -c2 * vw * u2 - c * uw                                 # Term 'b' in gyration formula
        d = 1 + 2 * c * uv + c2 * u2 * v2                         # Denominator with curvature scaling

        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)  # Gyrated result

    def inner(self, x, c, u, v=None, keepdim=False):
        """
        Compute the Riemannian inner product of two tangent vectors at point `x` on the Poincaré ball.

        This method generalizes the standard Euclidean inner product to the curved geometry of
        the Poincaré ball by scaling with the conformal factor λ(x)², which depends on the curvature [c]
        and the norm of point [x].

        When `v` is None, it computes the squared norm of vector `u`.

        Args:
            x (torch.Tensor): Point on the Poincaré ball where the tangent space is defined.
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).
            u (torch.Tensor): First tangent vector at point [x].
            v (torch.Tensor, optional): Second tangent vector at point [x]. Defaults to `u`.
            keepdim (bool): Whether to retain the last dimension after summation. Default: False.

        Returns:
            torch.Tensor: The Riemannian inner product ⟨u, v⟩_x in the tangent space at [x].
        """
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)                       # Conformal factor at point [x]
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)  # Scaled inner product

    def ptransp(self, x, y, u, c):
        """
        Perform parallel transport of a tangent vector `u` from point `x` to point `y` on the Poincaré ball manifold.

        Parallel transport moves a tangent vector along a geodesic from one point to another while preserving its
        geometric properties. In the Poincaré ball, this operation is implemented using the gyration operator and
        conformal factors at points [x]and [y].

        Args:
            x (torch.Tensor): Source point on the Poincaré ball where the vector `u` is defined.
            y (torch.Tensor): Target point on the Poincaré ball where the vector is transported.
            u (torch.Tensor): Tangent vector at point [x] to be transported.
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).

        Returns:
            torch.Tensor: The transported tangent vector at point [y], same shape as input vector `u`.
        """
        lambda_x = self._lambda_x(x, c)                # Conformal factor at source point [x]
        lambda_y = self._lambda_x(y, c)                # Conformal factor at target point [y]
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y  # Apply gyration and scale by λ ratios

    def ptransp_(self, x, y, u, c):
        """
        Perform parallel transport of a tangent vector `u` from point [x] to point `y` on the Poincaré ball manifold.

        This method is functionally identical to `ptransp`, and provides an alternate name for the same operation.
        It uses the gyration operator and conformal scaling via λ(x) and λ(y) to move the vector `u` along the geodesic
        from `x` to `y`, preserving its geometric properties in hyperbolic space.

        Args:
            x (torch.Tensor): Source point on the Poincaré ball where the vector `u` is defined.
            y (torch.Tensor): Target point on the Poincaré ball where the vector is transported.
            u (torch.Tensor): Tangent vector at point [x] to be transported.
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).

        Returns:
            torch.Tensor: The transported tangent vector at point [y], same shape as input vector `u`.
        """
        lambda_x = self._lambda_x(x, c)                # Conformal factor at source point [x]
        lambda_y = self._lambda_x(y, c)                # Conformal factor at target point [y]
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y  # Apply gyration and scale by λ ratios

    def ptransp0(self, x, u, c):
        """
        Perform parallel transport of a tangent vector `u` from point `x` to the origin on the Poincaré ball manifold.

        This method transports a vector from an arbitrary point [x] to the origin using a simplified rule specific to the origin.
        Unlike general parallel transport, this operation doesn't require gyration and uses only the conformal factor λ(x)
        to scale the vector appropriately for the change in geometry.

        Args:
            x (torch.Tensor): Source point on the Poincaré ball where the vector `u` is defined.
            u (torch.Tensor): Tangent vector at point [x] to be transported.
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).

        Returns:
            torch.Tensor: The transported tangent vector at the origin, same shape as input vector `u`.
        """
        lambda_x = self._lambda_x(x, c)                # Conformal factor at source point [x]
        return 2 * u / lambda_x.clamp_min(self.min_norm)  # Scale vector based on λ(x) for transport to origin

    def to_hyperboloid(self, x, c):
        """
        Map a point from the Poincaré ball to the hyperboloid (Lorentz) model of hyperbolic space.

        This method performs a coordinate transformation from the Poincaré ball representation to
        the hyperboloid (also known as the Lorentz or Minkowski model), which is another common
        embedding for hyperbolic geometry. The transformation preserves the intrinsic curvature
        and allows interoperability between different hyperbolic representations.

        The mapping follows the formula:
            H(x) = sqrt(K) * [K + ||x||², 2 * sqrt(K) * x] / (K - ||x||²)

        where K = 1 / c and [c] is the curvature of the Poincaré ball.

        Args:
            x (torch.Tensor): Point in the Poincaré ball, shape (n, d), where [d] is the dimensionality.
            c (float): Positive scalar curvature of the Poincaré ball (c > 0).

        Returns:
            torch.Tensor: The corresponding point in the hyperboloid model, shape (n, d+1).
        """
        K = 1. / c                          # Inverse curvature
        sqrtK = K ** 0.5                    # Square root of inverse curvature
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2  # Squared L2 norm of x

        # Construct the hyperboloid coordinates
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)

