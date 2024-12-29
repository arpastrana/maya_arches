from typing import List

from compas.geometry import Line
from compas.geometry import Plane
from compas.geometry import add_vectors
from compas.geometry import intersection_polyline_plane

from vaults import MayanVault


def create_slice_planes_by_block(vault: MayanVault, num_planes: int = 3) -> List[Line]:
    """
    Slices a vault horizontally, creating planar line slices.

    Notes
    ------
    This function will first separate the vault into separate wall, corbel and lintel blocks.
    The minimum number of slices is thus equal to 3.

    Afterwards,if the number of slices is greater than 3, the slices will be
    evenly distributed between the corbel and the wall.

    The lintel will remain as one block.
    """
    assert num_planes >= 3

    origin = [0.0, 0.0, 0.0]
    points = [
        origin,
        add_vectors(origin, [0.0, vault.wall_height, 0.0]),
        add_vectors(origin, [0.0, vault.wall_height + vault.corbel_height, 0.0]),
    ]

    return [Plane(point, [0.0, 1.0, 0.0]) for point in points]


def create_slice_planes(vault: MayanVault, num_planes: int = 2, max_height: float = None) -> List[Line]:
    """
    Slices a vault horizontally, creating planar line slices.

    Notes
    ------
    This function will first separate the lintel block from the rest of the vault.
    The minimum number of slices is thus equal to 2.

    If the number of slices is greater than 3, the slices will be
    evenly distributed between the base of the vault and the base of the lintel.
    """
    assert num_planes >= 2

    if not max_height:
        max_height = vault.wall_height + vault.corbel_height

    planes = []
    for i in range(num_planes):
        factor = i / (num_planes - 1)
        point = [0.0, factor * max_height, 0.0]
        planes.append(Plane(point, [0.0, 1.0, 0.0]))

    return planes


def slice_vault(vault: MayanVault, planes: List[Plane]) -> List[Line]:
    """
    Slices a vault horizontally, creating planar line slices.

    Notes
    ------
    This function will first separate the vault into separate wall, corbel and lintel blocks.
    The minimum number of slices is thus equal to 3.

    Afterwards,if the number of slices is greater than 3, the slices will be
    evenly distributed between the corbel and the wall.

    The lintel will remain as one block.
    """
    lines = []
    polyline = vault.polyline()
    for plane in planes:
        _points = intersection_polyline_plane(
            polyline,
            plane,
            expected_number_of_intersections=3
        )
        points = []
        for point in _points:
            if point not in points:
                points.append(point)

        assert len(points) == 2, f"Found {len(points)} points, I need only 2"
        points = sorted(points, key=lambda pt: pt[0])

        lines.append(Line(*points))

    return lines
