from compas.geometry import Polygon
from compas.geometry import Point
from compas.geometry import Line
from compas.geometry import Plane
from compas.geometry import centroid_polygon
from compas.geometry import cross_vectors
from compas.geometry import add_vectors
from compas.geometry import scale_vector
from compas.geometry import intersection_polyline_plane
from compas.geometry import distance_point_point
from compas.geometry import allclose
from compas.utilities import pairwise

from mayan_vaults.blocks.slicing import slice_vault
from mayan_vaults.blocks.slicing import create_slice_planes_by_block_horizontal
from mayan_vaults.blocks.slicing import create_slice_planes_by_block_vertical
from mayan_vaults.blocks.slicing import create_slice_planes_vertical


# ------------------------------------------------------------------------------
# Block
# ------------------------------------------------------------------------------

class Block:
    """
    A block is a 2D polygon representing a unit of a masonry arch.
    """
    def __init__(self, line_bottom: Line, line_top: Line, density: float):
        self.line_bottom = line_bottom
        self.line_top = line_top
        self.density = density

    def points(self) -> list[Point]:
        """
        The points of the block.
        """
        points = [
            self.line_bottom.start,
            self.line_bottom.end,
            self.line_top.end,
            self.line_top.start,
        ]

        return points

    def points_intrados(self) -> list[Point]:
        """
        The points of the intrados of the block.
        """
        return [self.line_bottom.end]

    def points_extrados(self) -> list[Point]:
        """
        The points of the extrados of the block.
        """
        return [self.line_bottom.start, self.line_top.start, self.line_top.end]

    def centroid(self) -> Point:
        """
        The centroid of the block.
        """
        return centroid_polygon(self.points())

    def plane(self) -> Plane:
        """
        The plane of the block.

        Notes
        -----
        The normal vector is computed as the average of the normals of 
        the bottom and top lines.
        """
        normal_bottom = cross_vectors(self.line_bottom.vector, [0.0, 0.0, 1.0])
        normal_top = cross_vectors(self.line_top.vector, [0.0, 0.0, 1.0])
        normal = scale_vector(add_vectors(normal_bottom, normal_top), 0.5)

        return Plane(self.centroid(), normal)

    def plane_line(self) -> Line:
        """
        The line of the plane of the block.
        """
        points = self.points()
        polyline = points + points[:1]

        points = intersection_polyline_plane(
            polyline,
            self.plane(),
            expected_number_of_intersections=2
        )
        assert len(points) == 2, f"Expected 2 points, got {len(points)}"

        return Line(*points)

    def polygon(self) -> Polygon:
        """
        The polygon of the block.
        """
        return Polygon(self.points())

    def area(self) -> float:
        """
        The area of the block.
        """
        return self.polygon().area

    def weight(self) -> float:
        """
        The weight of the block.
        """
        return self.area() * self.density

    def height(self) -> float:
        """
        The height of the block.
        """
        def calculate_midpoint(line: Line) -> Point:
            """
            Calculate the midpoint of a line.
            """
            return scale_vector(add_vectors(line.start, line.vector), 0.5)

        midpoint_top = calculate_midpoint(self.line_top)
        midpoint_bottom = calculate_midpoint(self.line_bottom)

        return distance_point_point(midpoint_top, midpoint_bottom)

    def __repr__(self):
        return f"Block(area={self.area():.2f}, height={self.height():.2f})"


# ------------------------------------------------------------------------------
# Block factory
# ------------------------------------------------------------------------------

def create_blocks(vault, num_blocks: int, density: float, slicing_method: int) -> list[Block]:
    """
    Create blocks from a given vault geometry.

    Notes
    -----
    The blocks are generated from the ground up by slicing the vault into 
    a prescribed number of blocks.

    The slicing method one of:
    - meta block horizontal (0)
    - meta block vertical (1)
    - linear range vertical (2)
    """
    # Create slice planes
    num_planes = num_blocks + 1

    if slicing_method == 0:
        planes = create_slice_planes_by_block_horizontal(vault, num_planes)    
    elif slicing_method == 1:
        planes = create_slice_planes_by_block_vertical(vault, num_planes)
    elif slicing_method == 2:
        planes = create_slice_planes_vertical(vault, num_planes)
    else:
        raise ValueError(f"Invalid slicing method id: {slicing_method}")

    # Slice the vault to create lines
    slice_lines = slice_vault(vault, planes)

    # Check slice lengths
    for i, line in enumerate(slice_lines):
        if slicing_method == 0:
            assert line.length <= vault.width / 2.0
        elif slicing_method == 1:
            assert line.length <= vault.height

    # Create blocks from slice lines
    blocks = []
    for i, (line_bottom, line_top) in enumerate(pairwise(slice_lines)):
        block = Block(line_bottom, line_top, density)
        
        # Check if the block has no area
        if allclose([block.area()], [0.0]):
            print(f"Block {i} has no area. Skipping...\n")
            continue

        blocks.append(block)

    # Check the number of blocks
    assert len(blocks) == num_blocks, f"Number of blocks does not match! Expected {num_blocks}, got {len(blocks)}"

    # Reverse the blocks to match the node order
    blocks = blocks[::-1]

    # Assign blocks to nodes
    block_dict = {}
    for i, block in enumerate(blocks):
        # Block keys start at 1 because node 0 is an origin node
        block_dict[i + 1] = block

    return block_dict
