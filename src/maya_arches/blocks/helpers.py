from typing import List

from compas.geometry import allclose


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