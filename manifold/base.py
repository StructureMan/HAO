"""Base manifold."""

from torch.nn import Parameter


class Manifold(object):
    """
    Abstract class to define operations on a manifold.
    """

    def __init__(self):
        """
        Constructor for the base Manifold class.

        Initializes common attributes used across different manifold implementations.
        This includes a small epsilon value (self.eps) used to prevent numerical instability
        during operations such as projections and distance calculations.

        Note:
            This class is abstract and should not be instantiated directly.
            It serves as a base class for specific manifold implementations.
        """
        super().__init__()
        self.eps = 10e-8  # Small epsilon value to avoid division by zero or other numerical issues

    def sqdist(self, p1, p2, c):
        """
        Compute the squared geodesic distance between two points on the manifold.

        This method is abstract and must be implemented by subclasses to define
        how distances are calculated in their specific geometry (e.g., Euclidean, Hyperbolic).

        Args:
            p1 (torch.Tensor): First point on the manifold.
            p2 (torch.Tensor): Second point on the manifold.
            c (float): Curvature parameter that defines the geometry of the manifold.

        Raises:
            NotImplementedError: This is an abstract method and should be overridden in a subclass.
        """
        raise NotImplementedError

    def egrad2rgrad(self, p, dp, c):
        """
        Convert Euclidean gradient to Riemannian gradient on the manifold.

        This method transforms a gradient computed in Euclidean space into the corresponding
        Riemannian gradient that respects the geometry of the manifold.
        
        In Euclidean space, this conversion has no effect (identity operation), but it becomes essential
        when working with curved manifolds like Hyperbolic or Spherical spaces.

        Args:
            p (torch.Tensor): Point on the manifold where the gradient is computed.
            dp (torch.Tensor): Euclidean gradient (vector in ambient space).
            c (float): Curvature parameter that defines the manifold's geometry.

        Raises:
            NotImplementedError: Must be implemented by subclasses representing specific manifolds.
        """
        raise NotImplementedError

    def proj(self, p, c):
        """
        Project a point onto the manifold.

        This method ensures that a given point lies on the manifold by projecting it
        if it's outside the valid space. This is especially important in curved geometries
        where operations are only defined for points that belong to the manifold.

        In Euclidean space, this operation has no effect since all points are valid,
        but it serves as a consistent interface across different manifold types.

        Args:
            p (torch.Tensor): Input point to be projected onto the manifold.
            c (float): Curvature parameter that defines the geometry of the manifold.

        Raises:
            NotImplementedError: Must be implemented by subclasses representing specific manifolds.
        """
        raise NotImplementedError

    def proj_tan(self, u, p, c):
        """
        Project a vector onto the tangent space at point p on the manifold.

        This method ensures that a given vector lies within the tangent space at point `p`,
        which is necessary for valid Riemannian operations such as exponential maps and gradients.

        In Euclidean space, all vectors are already in the tangent space (identity operation),
        but this method provides a consistent interface for other types of manifolds where projection is required.

        Args:
            u (torch.Tensor): Input vector to be projected.
            p (torch.Tensor): Point on the manifold at which the tangent space is defined.
            c (float): Curvature parameter that defines the geometry of the manifold.

        Raises:
            NotImplementedError: Must be implemented by subclasses representing specific manifolds.
        """
        raise NotImplementedError

    def proj_tan0(self, u, c):
        """
        Project a vector onto the tangent space at the origin of the manifold.

        This method ensures that a given vector lies within the tangent space at the origin,
        which is necessary for valid Riemannian operations such as exponential maps or inner products.

        In Euclidean space, all vectors are already in the tangent space (identity operation),
        but this method provides a consistent interface for other types of manifolds where projection is required.

        Args:
            u (torch.Tensor): Input vector to be projected.
            c (float): Curvature parameter that defines the geometry of the manifold.

        Raises:
            NotImplementedError: Must be implemented by subclasses representing specific manifolds.
        """
        raise NotImplementedError

    def expmap(self, u, p, c):
        """
        Compute the exponential map of a tangent vector at a point on the manifold.

        The exponential map moves a point `p` along the geodesic defined by the tangent vector `u`
        to produce a new point on the manifold. This operation generalizes vector addition
        to curved spaces and is fundamental in Riemannian geometry.

        In Euclidean space, this simplifies to standard vector addition: `p + u`.

        Args:
            u (torch.Tensor): Tangent vector at point `p`.
            p (torch.Tensor): Point on the manifold from which the exponential map is applied.
            c (float): Curvature parameter that defines the geometry of the manifold.

        Raises:
            NotImplementedError: Must be implemented by subclasses representing specific manifolds.
        """
        raise NotImplementedError

    def logmap(self, p1, p2, c):
        """
        Compute the logarithmic map from point p1 to point p2 on the manifold.

        The logarithmic map returns the tangent vector at point `p1` that represents the geodesic
        pointing from `p1` to `p2`. This operation is the inverse of the exponential map.

        In Euclidean space, this simplifies to simple vector subtraction: `p2 - p1`.

        Args:
            p1 (torch.Tensor): Source point on the manifold.
            p2 (torch.Tensor): Target point on the manifold.
            c (float): Curvature parameter that defines the geometry of the manifold.

        Raises:
            NotImplementedError: Must be implemented by subclasses representing specific manifolds.
        """
        raise NotImplementedError

    def expmap0(self, u, c):
        """
        Compute the exponential map of a tangent vector at the origin of the manifold.

        This method moves from the origin along the geodesic defined by the tangent vector `u`
        to produce a new point on the manifold. It generalizes vector addition starting from zero
        in Euclidean space.

        In Euclidean geometry, this simplifies to returning the vector `u` itself, since the origin
        is at zero and the space is flat.

        Args:
            u (torch.Tensor): Tangent vector at the origin.
            c (float): Curvature parameter that defines the geometry of the manifold.

        Raises:
            NotImplementedError: Must be implemented by subclasses representing specific manifolds.
        """
        raise NotImplementedError

    def logmap0(self, p, c):
        """
        Compute the logarithmic map of a point at the origin.

        This method returns the tangent vector at the origin that represents the geodesic
        from the origin to the given point `p`. It is the inverse of the exponential map at the origin.

        In Euclidean space, this simplifies to returning the point `p` itself, since the origin is at zero
        and the space is flat.

        Args:
            p (torch.Tensor): Point on the manifold.
            c (float): Curvature parameter that defines the geometry of the manifold.

        Raises:
            NotImplementedError: Must be implemented by subclasses representing specific manifolds.
        """
        raise NotImplementedError

    def mobius_add(self, x, y, c, dim=-1):
        """
        Perform Möbius addition of two points on the manifold.

        Möbius addition is a binary operation used in hyperbolic geometry that generalizes vector addition
        to curved spaces. It ensures that the result remains on the manifold, making it suitable for Riemannian optimization.

        In Euclidean space, this simplifies to standard vector addition: `x + y`.

        Args:
            x (torch.Tensor): First point on the manifold.
            y (torch.Tensor): Second point on the manifold.
            c (float): Curvature parameter that defines the geometry of the manifold.
            dim (int): The dimension along which the addition is applied (optional).

        Raises:
            NotImplementedError: Must be implemented by subclasses representing specific manifolds.
        """
        raise NotImplementedError

    def mobius_matvec(self, m, x, c):
        """
        Perform Möbius matrix-vector multiplication on the manifold.

        This operation generalizes standard matrix-vector multiplication to curved spaces,
        ensuring that the result remains on the manifold. It is particularly used in hyperbolic geometry.

        In Euclidean space, this simplifies to standard matrix-vector multiplication: `x @ m.T`.

        Args:
            m (torch.Tensor): Matrix to be multiplied with the vector.
            x (torch.Tensor): Point on the manifold (treated as a vector or batch of vectors).
            c (float): Curvature parameter that defines the geometry of the manifold.

        Raises:
            NotImplementedError: Must be implemented by subclasses representing specific manifolds.
        """
        raise NotImplementedError

    def init_weights(self, w, c, irange=1e-5):
        """
        Initialize random weights on the manifold.

        This method is responsible for initializing a given tensor `w` with small random values
        such that the resulting point lies on the manifold. It ensures stable initialization
        for use in Riemannian optimization.

        In Euclidean space, this typically involves simple uniform initialization within a small range.
        In curved spaces, additional constraints may be applied to ensure points lie within valid regions.

        Args:
            w (torch.Tensor): Tensor representing weights or points to be initialized on the manifold.
            c (float): Curvature parameter that defines the geometry of the manifold.
            irange (float): Range of the uniform distribution used for initialization.

        Raises:
            NotImplementedError: Must be implemented by subclasses representing specific manifolds.
        """
        raise NotImplementedError

    def inner(self, p, c, u, v=None, keepdim=False):
        """
        Compute the Riemannian inner product of two tangent vectors at a point on the manifold.

        This method calculates the inner product (dot product) of tangent vectors `u` and `v`
        at a given point `p`. The result respects the geometry defined by the curvature [c].

        If `v` is not provided, it defaults to `u`, computing the squared norm of `u`.

        In Euclidean space, this reduces to the standard dot product: `(u * v).sum(dim=-1)`.

        Args:
            p (torch.Tensor): Point on the manifold where the tangent vectors are defined.
            c (float): Curvature parameter that defines the geometry of the manifold.
            u (torch.Tensor): First tangent vector at point `p`.
            v (torch.Tensor, optional): Second tangent vector at point `p`. Defaults to `u` if not provided.
            keepdim (bool, optional): Whether to retain the last dimension in the output. Defaults to False.

        Returns:
            torch.Tensor: Inner product of `u` and `v` at point `p`.

        Raises:
            NotImplementedError: Must be implemented by subclasses representing specific manifolds.
        """

    def ptransp(self, x, y, u, c):
        """
        Perform parallel transport of a tangent vector from point x to point y on the manifold.

        Parallel transport moves a tangent vector `u` from the tangent space at point [x]
        to the tangent space at point [y] along a geodesic, preserving its geometric properties.

        In Euclidean space, this operation has no effect since all tangent spaces are identical,
        and vectors can be directly moved without any transformation: returns `u`.

        Args:
            x (torch.Tensor): Source point on the manifold.
            y (torch.Tensor): Target point on the manifold.
            u (torch.Tensor): Tangent vector at point [x] to be transported.
            c (float): Curvature parameter that defines the geometry of the manifold.

        Raises:
            NotImplementedError: Must be implemented by subclasses representing specific manifolds.
        """
        raise NotImplementedError

    def ptransp0(self, x, u, c):
        """
        Perform parallel transport of a tangent vector from the origin to a point [x] on the manifold.

        This method transports a tangent vector `u`, originally in the tangent space at the origin,
        to the tangent space at point `x` along a geodesic, preserving its geometric properties.

        In Euclidean space, this operation has no effect since all tangent spaces are identical,
        and vectors can be directly moved without any transformation: returns `u`.

        Args:
            x (torch.Tensor): Target point on the manifold where the vector should be transported.
            u (torch.Tensor): Tangent vector at the origin to be transported.
            c (float): Curvature parameter that defines the geometry of the manifold.

        Raises:
            NotImplementedError: Must be implemented by subclasses representing specific manifolds.
        """
        raise NotImplementedError


class ManifoldParameter(Parameter):
    """
    Subclass of torch.nn.Parameter for Riemannian optimization.
    """
    def __new__(cls, data, requires_grad, manifold, c):
        """
        Create a new instance of ManifoldParameter.

        This method overrides the __new__ method of the parent class (torch.nn.Parameter)
        to ensure that the data and gradient requirements are properly initialized.

        Args:
            data (torch.Tensor): The tensor data to initialize the parameter with.
            requires_grad (bool): Whether the parameter should require gradients.
            manifold (Manifold): The associated manifold used for Riemannian optimization.
            c (float): The curvature value associated with the manifold.

        Returns:
            ManifoldParameter: A new instance of ManifoldParameter with initialized data and gradient settings.
        """
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold, c):
        """
        Initialize a ManifoldParameter instance.

        This constructor associates the parameter with a specific manifold and curvature value,
        enabling it to be used in Riemannian optimization methods that respect the geometry of the space.

        Args:
            data (torch.Tensor): The tensor data to wrap as a parameter.
            requires_grad (bool): Whether the parameter should require gradients.
            manifold (Manifold): The manifold on which this parameter lies (e.g., Euclidean, Hyperbolic).
            c (float): The scalar curvature of the manifold (controls the geometry).
        """
        self.c = c           # Curvature of the manifold (e.g., 0 for Euclidean, <0 for Hyperbolic)
        self.manifold = manifold  # Reference to the manifold for geometric operations

    def __repr__(self):
        """
        Return a string representation of the ManifoldParameter.

        This method overrides the default representation to include information about
        the manifold and curvature it is associated with, making debugging and inspection more intuitive.

        Returns:
            str: Human-readable string describing the parameter and its manifold context.
        """
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()
