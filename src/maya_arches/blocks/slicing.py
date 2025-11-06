from typing import List
from typing import TYPE_CHECKING

from compas.geometry import Line
from compas.geometry import Plane
from compas.geometry import Point
from compas.geometry import Vector
from compas.geometry import Translation
from compas.geometry import intersection_polyline_plane
from compas.geometry import scale_vector
from compas.geometry import add_vectors

from maya_arches.blocks import estimate_num_objects_percentages
from maya_arches.blocks import round_numbers_integer_sum

if TYPE_CHECKING:
    from maya_arches.arches import Arch


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def create_planes_linrange(origin_start: Point, direction: Vector, distance: float, num_planes: int) -> List[Plane]:
    """
    Creates a list of planes linearly distributed between a start and end plane.
    """
    assert num_planes >= 2

    planes = []
    for i in range(num_planes):
        factor = distance * (i / (num_planes - 1))
        origin = add_vectors(origin_start, scale_vector(direction, factor))

        planes.append(Plane(origin, direction))

    return planes


# ------------------------------------------------------------------------------
# Horizontal slicing
# ------------------------------------------------------------------------------

def create_slice_planes_by_block_horizontal(arch: "Arch", num_planes: int = 4) -> List[Line]:
    """
    Slices an arch horizontally, creating planar line slices per block.

    Notes
    ------
    This function will first separate the arch into separate wall, corbel and lintel blocks.
    The minimum number of slices is thus equal to 4.

    Afterwards, if the number of slices is greater than 4, the slices will be
    evenly distributed between the corbel and the wall.

    The lintel will remain as one block.
    """
    num_planes_min = 4
    assert num_planes >= num_planes_min, f"The number of planes must be greater than or equal to {num_planes_min}"

    # Create planes
    heights = [arch.wall_height, arch.corbel_height, arch.lintel_height]
    num_meta_blocks = len(heights)

    # Minimum number of planes is 2 per segment
    num_planes_extra = num_planes - num_planes_min

    # Estimate number of planes per segment
    # TODO: Squaring the heights is hacky, find a better solution!
    weights = [height ** 2 for height in heights]
    num_planes_per_segment = estimate_num_objects_percentages(weights, num_planes_extra)
    count_extra = round_numbers_integer_sum(num_planes_per_segment)

    # Add extra number of planes to the minimum
    num_planes_per_segment_min = 2
    num_planes_per_segment = [num_planes_per_segment_min + c_extra for c_extra in count_extra]

    # Create planes
    planes = []
    planes.append(Plane([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]))

    _height_previous = 0.0
    for i in range(num_meta_blocks):
        _num_planes = num_planes_per_segment[i]
        _height = heights[i]

        planes_meta_block = create_planes_linrange(
            Point(0.0, 0.0, 0.0),
            Vector(0.0, 1.0, 0.0),
            _height,
            _num_planes
        )

        # Translate the planes upwards
        # TODO: Remove this once the planes linrange is fixed
        if i > 0:
            _height_previous = heights[i - 1] + _height_previous
            T = Translation.from_vector([0.0, _height_previous, 0.0])
            planes_meta_block = [plane.transformed(T) for plane in planes_meta_block]

        planes_meta_block.pop(0)
        planes.extend(planes_meta_block)

    assert len(planes) == num_planes

    return planes


# ------------------------------------------------------------------------------
# Vertical slicing
# ------------------------------------------------------------------------------

def create_slice_planes_by_block_vertical(arch: "Arch", num_planes: int = 3) -> List[Line]:
    """
    Slices an arch vertically, creating planar line slices per block.

    Notes
    ------
    This function will first separate the arch into separate wall and span blocks.
    The minimum number of slices is thus equal to 2.

    Afterwards, if the number of slices is greater than 2, the slices will be
    evenly distributed between the base of the arch and the base of the span.

    The lintel will be sliced as well.
    """
    num_planes_min = 3
    assert num_planes >= num_planes_min, f"The number of planes must be greater than or equal to {num_planes_min}"

    # Create planes
    widths = [arch.wall_width, arch.span_half]
    num_meta_blocks = len(widths)

    # Minimum number of planes is 2 per segment
    num_planes_extra = num_planes - num_planes_min

    # Estimate number of planes per segment
    weights = [width for width in widths]
    num_planes_per_segment = estimate_num_objects_percentages(weights, num_planes_extra)
    count_extra = round_numbers_integer_sum(num_planes_per_segment)

    # Add extra number of planes to the minimum
    num_planes_per_segment_min = 2
    num_planes_per_segment = [num_planes_per_segment_min + c_extra for c_extra in count_extra]

    # Create planes
    planes = []
    planes.append(Plane([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))

    _width_previous = 0.0
    for i in range(num_meta_blocks):
        _num_planes = num_planes_per_segment[i]
        _width = widths[i]

        planes_meta_block = create_planes_linrange(
            Point(0.0, 0.0, 0.0),
            Vector(1.0, 0.0, 0.0),
            _width,
            _num_planes
        )

        # TODO: Remove this once the planes linrange is fixed
        # Translate the planes to the right
        if i > 0:
            _width_previous = widths[i - 1] + _width_previous
            T = Translation.from_vector([_width_previous, 0.0, 0.0])
            planes_meta_block = [plane.transformed(T) for plane in planes_meta_block]

        planes_meta_block.pop(0)
        planes.extend(planes_meta_block)

    assert len(planes) == num_planes, f"Number of planes does not match: {len(planes)} != {num_planes}"

    return planes


def create_slice_planes_vertical(arch: "Arch", num_planes: int = 2) -> List[Line]:
    """
    Slices an arch vertically, creating uniformly spaced planar line slices per block.
    """
    num_planes_min = 2
    assert num_planes >= num_planes_min, f"The number of planes must be greater than or equal to {num_planes_min}"

    # Create planes
    planes = create_planes_linrange(
        Point(0.0, 0.0, 0.0),
        Vector(1.0, 0.0, 0.0),
        arch.width * 0.4999,  # NOTE: This is a hack to avoid the line being too close to the edge of the arch
        num_planes
    )

    assert len(planes) == num_planes, f"Number of planes does not match: {len(planes)} != {num_planes}"

    return planes


# ------------------------------------------------------------------------------
# Caller functions
# ------------------------------------------------------------------------------

def slice_arch(arch: "Arch", planes: List[Plane]) -> List[Line]:
    """
    Slices an arch with a sequence of planes.
    """
    lines = []
    polyline = arch.polyline()

    for i, plane in enumerate(planes):

        _points = intersection_polyline_plane(
            polyline,
            plane,
            expected_number_of_intersections=3,
        )

        points = []
        for point in _points:
            if point not in points:
                points.append(point)

        # Sort the points by coordinate y
        points = sorted(points, key=lambda pt: pt[1])

        # TODO: This is a hack, find a better solution!
        if len(points) > 2:
            print(f"Found {len(points)} points with plane {i}. Creating {len(points) - 2} extra lines.")
            point_base = points.pop()
            for point in points:
                lines.append(Line(point, point_base))
        elif len(points) == 1:
            print(f"Found 1 point with plane {i}. Creating 1 line by duplicating the point.")
            point = points[0]
            lines.append(Line(point, point))
        else:
            lines.append(Line(*points))

    return lines
