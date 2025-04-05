from math import sqrt
from math import pi
from math import cos
from math import sin

from compas.geometry import Point

from mayan_vaults.vaults.vaults import Vault


# ------------------------------------------------------------------------------
# Elliptical vault
# ------------------------------------------------------------------------------

class EllipticalVault(Vault):
    """
    An elliptical vault.
    """
    def __init__(self, height: float, width: float, thickness: float, **kwargs):
        super().__init__()
        self._height = height
        self._width = width
        self._thickness = thickness

        self._check_thickness()

    def _check_thickness(self):
        """
        Check the thickness of the vault.
        """
        if self.thickness > self.height:
            raise ValueError("The thickness of the vault cannot be greater than the height.")
        if self.thickness > self.width:
            raise ValueError("The thickness of the vault cannot be greater than the width.")

    @property
    def height(self) -> float:
        """
        The height of the vault.
        """
        return self._height

    @property
    def width(self) -> float:
        """
        The width of the vault.
        """
        return self._width

    @property    
    def thickness(self) -> float:
        """
        The thickness of the vault. 

        The thickness is exact at the base and the top of the vault due to the
        simpleoffset applied to the points of the extrados and intrados.
        """
        return self._thickness

    @property
    def support_width(self) -> float:
        """
        The width of the support.
        """
        return self.thickness

    @property
    def span(self) -> float:
        """
        The span of the vault.
        """
        return 2 * (self.width - self.thickness)

    def points(self, num_points: int = 100) -> list[Point]:
        """
        The points of the vault.
        """
        radius_1 = self.height
        radius_2 = self.width
        points_extrados = points_elliptical_arc(radius_1, radius_2, num_points)

        radius_3 = radius_1 - self.thickness
        radius_4 = radius_2 - 2 * self.thickness
        points_intrados = points_elliptical_arc(radius_3, radius_4, num_points)

        def shift_points_x(points, offset):
            return [[point[0] + offset, point[1], 0.0] for point in points]

        shift = radius_2 * 0.5
        points_extrados = shift_points_x(points_extrados, shift)
        points_intrados = shift_points_x(points_intrados, shift)

        return points_extrados + points_intrados[::-1]

    def __str__(self) -> str:
        """
        Return a string representation of the elliptical vault.
        """
        params = {'thickness': self.thickness}

        return super().__str__(params)


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def points_elliptical_arc(
        height: float, 
        width: float, 
        num_points: int        
        ) -> list[list[float]]:
    """Generate points on a planar elliptical arc.
    
    Parameters
    ----------
    height : float
        The height of the parabolic arc.
    width : float
        The width of the parabolic arc.
    num_points : int
        The number of points to generate.
    
    Returns
    -------
    list[list[float]]
        List of [x, y, z] coordinates of points on the arc.
        First point is [0, 0, 0].
        Last point is [width, height, 0].
    """
    if num_points < 2:
        raise ValueError("Number of points must be at least 2")
    
    points = []    
    for theta in linspace(pi, pi * 0.5, num_points):
        x = 0.5 * width * cos(theta)
        y = height * sin(theta)        
        point = [x, y, 0.0]        
        points.append(point)

    return points


def points_elliptical_arc_offset(
        height: float, 
        width: float, 
        offset: float,
        num_points: int,
        ) -> list[list[float]]:
    """Generate points on a planar elliptical arc.
    
    Parameters
    ----------
    height : float
        The height of the parabolic arc.
    width : float
        The width of the parabolic arc.
    offset : float
        The offset of the arc.
    num_points : int
        The number of points to generate.
    
    Returns
    -------
    list[list[float]]
        List of [x, y, z] coordinates of points on the arc.
        First point is [0, 0, 0].
        Last point is [width - offset, height - offset, 0].
    """
    if num_points < 2:
        raise ValueError("Number of points must be at least 2")
        
    points = []    
    for theta in linspace(pi, pi * 0.5, num_points):
        # Compute coordinates
        x = 0.5 * width * cos(theta)
        y = height * sin(theta)

        # Compute unit normals
        nx = width * cos(theta)
        ny = height * sin(theta)
        norm = sqrt(nx**2 + ny**2)
        nx /= norm
        ny /= norm

        # Offset coordinates
        x = x + offset * nx
        y = y + offset * ny      
        
        point = [x, y, 0.0]        
        points.append(point)

    return points


def linspace(start, stop, num, endpoint=True):
    """
    Return evenly spaced numbers over a specified interval.

    Returns a list of evenly spaced numbers over the specified interval.

    Parameters
    ----------
    start : float
        The start of the interval.
    stop : float
        The end of the interval.
    num : int
        The number of points to generate.
    endpoint : bool, optional
        If True, the endpoint is included.
    """
    if num <= 0:
        return []
    if num == 1:
        return [stop if endpoint else start]

    step = (stop - start) / (num - 1) if endpoint else (stop - start) / num

    return [start + i * step for i in range(num)]
