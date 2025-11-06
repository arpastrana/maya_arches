from compas.geometry import Point
from math import pi
from math import cos
from math import sin

from maya_arches.arches import Arch


# ------------------------------------------------------------------------------
# Circular arch
# ------------------------------------------------------------------------------

class CircularArch(Arch):
    """
    A half-circular arch.
    """
    def __init__(self, radius: float, thickness: float, **kwargs):
        super().__init__()
        self.radius = radius
        self._thickness = thickness

        self._check_thickness()

    def _check_thickness(self):
        """
        Check the thickness of the arch.
        """
        if self.thickness > self.radius:
            raise ValueError("Arch thickness cannot be greater than the radius!")

    @property
    def height(self) -> float:
        """
        The height of the arch.
        """
        return self.radius

    @property
    def width(self) -> float:
        """
        The width of the arch.
        """
        return self.radius * 2.0

    @property    
    def thickness(self) -> float:
        """
        The thickness of the arch.
        """
        return self._thickness

    @property
    def support_width(self) -> float:
        """
        The width of the arch's support.
        """
        return self.thickness

    @property
    def span(self) -> float:
        """
        The span of the arch.
        """
        return 2 * (self.radius - self.thickness)
    
    def points(self, num_points: int = 100) -> list[Point]:
        """
        The points of the arch.
        """        
        radius_extrados = self.radius
        points_extrados = points_circle_arc_half(radius_extrados, num_points)    

        radius_intrados = radius_extrados - self.thickness
        points_intrados = points_circle_arc_half(radius_intrados, num_points, self.thickness)

        return points_extrados + points_intrados[::-1]

    def __str__(self) -> str:
        """Return a string description of this arch."""
        params = {'radius': self.radius, 'thickness': self.thickness}

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