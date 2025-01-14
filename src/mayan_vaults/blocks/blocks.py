from compas.geometry import Translation
from compas.geometry import Polygon
from compas.geometry import Point
from compas.geometry import Line
from compas.geometry import Plane

from mayan_vaults.blocks.slicing import slice_vault
from mayan_vaults.blocks.slicing import create_slice_planes
from mayan_vaults.blocks.slicing import create_slice_planes_by_block


class Block:
    """
    A block is a 2D polygon representing a unit of a vault.
    """
    def __init__(self, plane: Plane, line_bottom: Line, line_top: Line, density: float):
        self.plane = plane
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
        return self.line_top.midpoint.y - self.line_bottom.midpoint.y
    
    def __repr__(self):
        return f"Block(area={self.area():.2f}, height={self.height():.2f})"


def create_blocks(vault, num_blocks: int, density: float, slicing_method: int) -> list[Block]:
    """
    Create blocks from a given vault geometry.

    Notes
    -----
    The blocks are generated from the ground up by slicing the vault into 
    a prescribed number of blocks.

    The slicing method can be either by meta block (0) or by height (1).
    """
    # Create slice planes
    if slicing_method == 0:
        planes = create_slice_planes_by_block(vault, num_blocks)
    elif slicing_method == 1:    
        max_height = vault.wall_height + vault.corbel_height
        print(f"Max height for slicing: {max_height}")
        planes = create_slice_planes(vault, num_blocks, max_height)
    else:
        raise ValueError(f"Invalid slicing method: {slicing_method}")

    # Slice the vault to create lines
    slice_lines = slice_vault(vault, planes)

    # Insert a line at the top of the vault
    T = Translation.from_vector([0.0, vault.lintel_height, 0.0])
    slice_line_top = slice_lines[-1].transformed(T)
    slice_lines.append(slice_line_top)

    # Check slice lengths
    for i, line in enumerate(slice_lines):
        assert line.length <= vault.width / 2.0

    # Create blocks from slice lines
    blocks = []
    for i in range(num_blocks):
        plane = planes[i]
        line_bottom = slice_lines[i]
        line_top = slice_lines[i + 1]

        block = Block(plane, line_bottom, line_top, density)
        blocks.append(block)

    assert len(blocks) == num_blocks, "Number of blocks does not match!"

    # Reverse the blocks to match the node order
    blocks = blocks[::-1]

    # Assign blocks to nodes
    block_dict = {}
    block_dict[0] = blocks[0]

    for i, block in enumerate(blocks):
        block_dict[i + 1] = block

    return block_dict
