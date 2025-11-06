from compas.geometry import add_vectors
from compas.geometry import Point

from maya_arches.arches import Arch


# ------------------------------------------------------------------------------
# MayaArch
# ------------------------------------------------------------------------------

class MayaArch(Arch):
    """
    One half of the 2D geometry of a Maya arch.
    """
    def __init__(
            self, 
            height: float, 
            width: float, 
            wall_height: float, 
            wall_width: float, 
            lintel_height: float,
            **kwargs
        ):
        super().__init__()

        self._height = height
        self._width = width

        self.wall_height = wall_height
        self.wall_width = wall_width
        self.lintel_height = lintel_height

        self._check_width()
        self._check_height()

    @property
    def height(self) -> float:
        """
        The height of the arch.
        """
        return self._height

    @property
    def width(self) -> float:
        """
        The width of the arch.
        """
        return self._width

    @property    
    def thickness(self) -> float:
        """
        The thickness of the arch, considered as the lintel height.
        """
        return self.lintel_height

    @property
    def support_width(self) -> float:
        """
        The width of the arch's support.
        """
        return self.wall_width

    @property
    def span(self) -> float:
        """
        The span of the arch.
        """
        return self.width - 2 * self.wall_width
    
    @property
    def corbel_height(self) -> float:
        """
        The height of the corbel.
        """
        return self.height - self.wall_height - self.lintel_height
        
    def points(self) -> list[Point]:
        """
        The points.
        """
        p0 = [0.0, 0.0, 0.0]
        p1 = add_vectors(p0, [self.wall_width, 0.0, 0.0])
        p2 = add_vectors(p1, [0.0, self.wall_height, 0.0])

        p3 = add_vectors(p2, [self.span / 2.0, self.corbel_height, 0.0])
        p4 = add_vectors(p3, [0.0, self.lintel_height, 0.0])
        p5 = add_vectors(p0, [0.0, self.height, 0.0])

        return [p0, p1, p2, p3, p4, p5]

    def _check_width(self) -> None:
        """
        Checks if the width of the arch makes sense.
        """
        assert self.width > self.wall_width, "The width of the wall is greater than the arch's!"

    def _check_height(self) -> None:
        """
        Checks if the height of the arch makes sense.
        """
        msg = "The height of the arch is smaller than the wall's and the lintel's combined!"
        assert self.height >= (self.wall_height + self.lintel_height), msg

        if self.lintel_height > self.wall_height:
            print("\nWarning! The lintel is deeper than the walls")

    def __str__(self) -> str:
        """Return a string description of this arch."""
        params = {            
            'wall_height': self.wall_height,
            'wall_width': self.wall_width,
            'corbel_height': self.corbel_height,
            'lintel_height': self.lintel_height,            
        }
        return super().__str__(params)
