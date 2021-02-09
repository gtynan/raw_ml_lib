import pytest
import numpy as np

from src.distance_functions import hamming_distance, absolute_difference, euclidean_distance

@pytest.mark.parametrize('x1, x2, res', [
    (np.array(["Female", "Irish"]), np.array(["Male", "Irish"]), 1),
    (np.array(["Female", "Irish"]), np.array(["Male", "Italian"]), 2),
    (np.array(["Male", "Irish"]), np.array(["Male", "Italian"]), 1),
])
def test_hamming_distance(x1, x2, res):
    assert hamming_distance(x1, x2) == res


@pytest.mark.parametrize('x1, x2, res', [
    (np.array([2.5, 6]), np.array([3.75, 8]), 3.25),
    (np.array([2.5, 6]), np.array([2.25, 5.5]), 0.75),
    (np.array([3.75, 8]), np.array([2.25, 5.5]), 4),
])
def test_absolute_difference(x1, x2, res):
    assert absolute_difference(x1, x2) == res


@pytest.mark.parametrize('x1, x2, res', [
    (np.array([3.25, 8.25]), np.array([4.75, 6.25]), 2.5),
    (np.array([3.25, 8.25]), np.array([2.75, 7.5]), 0.9),
])
def test_euclidean_distance(x1, x2, res):
    pytest.approx(euclidean_distance(x1, x2), res)
