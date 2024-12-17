from compas.geometry import add_vectors
from compas.geometry import Polyline
from compas.geometry import Polygon


class MayanVault:
    pass


class HalfMayanVault2D(MayanVault):
    """
    One half of the 2D geometry of a Mayan vault.
    """
    def __init__(self, height, width, wall_height, wall_width, lintel_height):
        self.height = height
        self.width = width
        self.wall_height = wall_height
        self.wall_width = wall_width
        self.lintel_height = lintel_height

        self._check_width()
        self._check_height()

    @property
    def span(self):
        """
        """
        return self.width - 2 * self.wall_width

    @property
    def corbel_height(self):
        """
        """
        return self.height - self.wall_height - self.lintel_height

    def points(self):
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

    def polyline(self):
        """
        The silhouette of the vault as a polyline.
        """
        points = self.points()
        return Polyline(points + points[:1])

    def polygon(self):
        """
        The silhouette of the vault as a polygon.
        """
        return Polygon(self.points())

    def _check_width(self):
        """
        Checks if the width of the vault makes sense.
        """
        assert self.width > self.wall_width, "The width of the wall is greater than the vault's!"

    def _check_height(self):
        """
        Checks if the height of the vault makes sense.
        """
        msg = "The height of the vault is smaller than the wall's and the lintel's combined!"
        assert self.height > (self.wall_height + self.lintel_height), msg

        if self.lintel_height > self.wall_height:
            print("\nWarning! The lintel is taller than the walls")
