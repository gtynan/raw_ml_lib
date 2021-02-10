import pytest
import numpy as np

from src.entropy import entropy, information_gain, get_max_information_gain_col

@pytest.mark.parametrize('arr, res', [([1, 1, 1], 0), ([1, 1, 1, 0, 0, 0], 1)])
def test_entropy(arr, res):
    assert entropy(arr) == res


@pytest.mark.parametrize('arr, y, res', [
    (np.array([0,0,1,1,1,1,2,2,2,2,2,2]), np.array([0,0,1,1,1,1,1,1,0,0,0,0]), 0.541),
    (np.array([0,0,1,1,1,1,2,2,3,3,3,3]), np.array([0,1,0,1,0,1,0,1,0,1,0,1]), 0),
])
def test_information_gain(arr, y, res):
    pytest.approx(information_gain(arr, y), res)

def test_get_max_information_gain_col():
    # 4 rows 2 cols
    X = np.array([[0,0,0,1], [0,0,1,1]]).T
    # 4 rows 1 col
    y = np.array([0,0,1,1])
    assert get_max_information_gain_col(X, y) == 1
