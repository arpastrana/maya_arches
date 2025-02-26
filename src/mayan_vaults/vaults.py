from compas.geometry import add_vectors
from compas.geometry import Polyline
from compas.geometry import Polygon

from mayan_vaults.blocks import create_blocks


class MayanVault:
    """
    A base class for Mayan vaults.
    """
    def __str__(self):
        """Return a string representation of the MayanVault object."""
        params = {
            'height': self.height,
            'width': self.width,
            'wall_height': self.wall_height,
            'wall_width': self.wall_width,
            'lintel_height': self.lintel_height,
            'span': self.span,
            'corbel_height': self.corbel_height,
            'num_blocks': len(self.blocks)
        }

        params_formatted = []
        for key, value in params.items():
            if isinstance(value, float):
                value = f"{value:.2f}"
            params_formatted.append(f"\t{key}={value}")

        return f"{self.__class__.__name__}(\n" + "\n".join(params_formatted) + "\n)"


class HalfMayanVault2D(MayanVault):
    """
    One half of the 2D geometry of a Mayan vault.
    """
    # TODO: Implement print out string representation method!
    def __init__(self, height, width, wall_height, wall_width, lintel_height):
        self.height = height
        self.width = width
        self.wall_height = wall_height
        self.wall_width = wall_width
        self.lintel_height = lintel_height

        self._check_width()
        self._check_height()

        self.blocks = {}

    @property
    def span(self):
        """
        The span of the vault.
        """
        return self.width - 2 * self.wall_width

    @property
    def corbel_height(self):
        """
        The height of the corbel.
        """
        return self.height - self.wall_height - self.lintel_height
        
    def weight(self):
        """
        The weight of the vault.

        Notes
        -----
        The weight of block 0 is excluded from the weight calculation
        because it is a duplicate of block 1.
        """
        key_first = 0
        return sum(block.weight() for key, block in self.blocks.items() if key != key_first)

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

    def blockify(self, num_blocks: int, density: float, slicing_method: int) -> None:
        """
        Create blocks from the vault. The lintel roof won't be sliced.
        """
        self.blocks = create_blocks(self, num_blocks, density, slicing_method)

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


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def create_vault(
        height: float,
        width: float,
        wall_height: float,
        wall_width: float,
        lintel_height: float,
        num_blocks: int,
        slicing_method: int,
        block_density: float
    ) -> MayanVault:
    """
    Create a blockified Mayan vault.
    """
    vault = HalfMayanVault2D(
        height,
        width,
        wall_height,
        wall_width,
        lintel_height
    )

    vault.blockify(num_blocks, block_density, slicing_method)

    return vault