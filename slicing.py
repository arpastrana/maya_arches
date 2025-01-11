from typing import List

from compas.geometry import Line
from compas.geometry import Plane
from compas.geometry import Translation
from compas.geometry import add_vectors
from compas.geometry import intersection_polyline_plane
from compas.geometry import allclose

from vaults import MayanVault


def round_numbers_integer_sum(xs: List[float]) -> List[int]:
    """
    Convert a series of floats to a sequence of integers, while preserving the total sum. 

    Notes
    -----
    The input sequence of floats must add up to an integer.

    Source: https://stackoverflow.com/questions/44737874/rounding-floats-while-maintaining-total-sum-equal
    """
    N = sum(xs)
    Rs = [int(x) for x in xs]
    K = N - sum(Rs)
    assert allclose([K], [round(K)]), f"The input list does not add up to an integer. This is invalid. {K=} vs. {round(K)=}"

    if allclose([K], [0.0]):
        return Rs

    K = round(K)
    fs = [x - int(x) for x in xs]
    sorted_vals = sorted([(e, i) for i, e in enumerate(fs)], reverse=True)    

    counter = 0
    indices = []
    for _, i in sorted_vals:
        if counter < K:
            indices.append(i)
        counter += 1
        
    ys = [R + 1 if i in indices else R for i, R in enumerate(Rs)]

    assert allclose([N], [sum(ys)]), f"The sum of the rounded sequence is different from the sum of the inputs. Target: {N} vs. Current: {sum(ys)}"

    return ys


def estimate_num_objects_percentages(values: List[float], num_objects: int, total: float = None) -> List[float]:
    """
    Assign a number of objects based on a sequence of values.
    """
    if not total:
        total = sum(values)

    return [num_objects * value / total for value in values]


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
    num_planes_min = 3
    assert num_planes >= num_planes_min, f"The number of planes must be greater than or equal to {num_planes_min}"

    # Create planes
    # Heights: wall height, corbel height, lintel will always be one block
    heights = [vault.wall_height, vault.corbel_height]
    num_meta_blocks = len(heights)

    # Minimum number of planes is 2 per segment
    num_planes_extra = num_planes - num_planes_min

    # Estimate number of planes per segment
    weights = [height ** 2 for height in heights]
    num_planes_per_segment = estimate_num_objects_percentages(weights, num_planes_extra)
    count_extra = round_numbers_integer_sum(num_planes_per_segment)

    print(f"\t{num_planes_extra=}")
    print(f"\t{heights=}")
    print(f"\t{num_planes_per_segment=}")
    print(f"\t{count_extra=}")

    # Add extra number of planes to the minimum
    num_planes_per_segment_min = 2
    num_planes_per_segment = [num_planes_per_segment_min + c_extra for c_extra in count_extra]

    # Create planes    
    planes = []
    planes.append(Plane([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]))

    for i in range(num_meta_blocks):
        _num_planes = num_planes_per_segment[i]
        _height = heights[i]
        planes_meta_block = create_slice_planes(vault, _num_planes, _height)

        # Translate the planes upwards
        if i > 0:
            _height_previous = heights[i - 1]
            T = Translation.from_vector([0.0, _height_previous, 0.0])
            planes_meta_block = [plane.transformed(T) for plane in planes_meta_block]

        planes_meta_block.pop(0)
        planes.extend(planes_meta_block)

    assert len(planes) == num_planes

    return planes


def create_slice_planes(vault: MayanVault, num_planes: int = 2, max_height: float = None) -> List[Line]:
    """
    Slices a vault horizontally, creating planar line slices.

    Notes
    ------
    This function will first separate the lintel block from the rest of the vault.
    The minimum number of slices is thus equal to 2.

    If the number of slices is greater than 2, the slices will be
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
