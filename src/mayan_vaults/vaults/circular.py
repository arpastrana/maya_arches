from compas.geometry import Point
from math import pi, cos, sin

from mayan_vaults.blocks import create_blocks
from mayan_vaults.vaults.vaults import Vault


# ------------------------------------------------------------------------------
# Circular vault
# ------------------------------------------------------------------------------

class CircularVault(Vault):
    """
    A circular vault.
    """
    def __init__(self, radius: float, thickness: float, **kwargs):
        super().__init__()
        self.radius = radius
        self._thickness = thickness

        self._check_thickness()

    def _check_thickness(self):
        """
        Check the thickness of the vault.
        """
        if self.thickness > self.radius:
            raise ValueError("The thickness of the vault cannot be greater than the radius.")

    @property
    def height(self) -> float:
        """
        The height of the vault.
        """
        return self.radius

    @property
    def width(self) -> float:
        """
        The width of the vault.
        """
        return self.radius * 2.0

    @property    
    def thickness(self) -> float:
        """
        The thickness of the vault.
        """
        return self._thickness

    @property
    def span(self) -> float:
        """
        The span of the vault.
        """
        return 2 * (self.radius - self.thickness)
    
    @property
    def span_half(self) -> float:
        """
        The span of the vault.
        """
        return self.span / 2.0

    def points(self, num_points: int = 101) -> list[Point]:
        """
        The points of the vault.
        """
        assert num_points % 2 == 1, "Number of points must be odd"

        radius_extrados = self.radius
        points_extrados = points_circle_arc_half(radius_extrados, num_points)    

        radius_intrados = radius_extrados - self.thickness
        points_intrados = points_circle_arc_half(radius_intrados, num_points, self.thickness)

        return points_extrados + points_intrados[::-1]
    
    def blockify(self, num_blocks: int, density: float, slicing_method: int) -> None:
        """
        Create blocks from the vault.
        """        
        self.blocks = create_blocks(self, num_blocks, density, slicing_method)

    def __str__(self) -> str:
        """Return a string representation of the CircularVault object."""
        params = {            
            'radius': self.radius,
            'thickness': self.thickness,
            'num_blocks': len(self.blocks)
        }
        return super().__str__(params)


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def points_circle_arc_half(radius: float, num_points: int, offset_x: float = 0.0) -> list[list[float]]:
    """Generate points on a planar circular arc.
    
    Parameters
    ----------
    radius : float
        The radius of the circle.
    num_points : int
        The number of points to generate.
    
    Returns
    -------
    list[list[float]]
        List of [x, y, z] coordinates of points on the arc.
        First point is [0, 0, 0].
        Last point is [radius, radius, 0].
    """
    if num_points < 2:
        raise ValueError("Number of points must be at least 2")
    
    points = []
    # Generate angles from pi to 0 (counter-clockwise from -x to +x)
    for i in range(num_points):
        t = pi * 0.5 * (1 - i / (num_points - 1))
        x = radius * (1 - cos(t))
        y = radius * sin(t)
        points.append([x + offset_x, y, 0.0])
    
    return points[::-1]