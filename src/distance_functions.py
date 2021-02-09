import numpy as np


def hamming_distance(x1: np.ndarray, x2: np.ndarray) -> int:
    """Global distance function which is the sum of the overlap differences across 
    all features - i.e. number of features on which two examples disagree.

    Args:
        x1 (np.ndarray): array 1
        x2 (np.ndarray): array 2

    Returns:
        int: hamming distance between two arrays
    """
    assert isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray)
    return (x1 != x2).sum()


def absolute_difference(x1: np.ndarray, x2: np.ndarray) -> float:
    """For numeric data, we can calculate absolute value of the 
    difference between values for a feature.

    Args:
        x1 (np.ndarray): array 1
        x2 (np.ndarray): array 2

    Returns:
        float: absolute difference
    """
    assert isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray)
    return np.absolute(x1 - x2).sum()


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """straight line distance between two points in a Euclidean 
    coordinate space - e.g. a feature space.

    Args:
        x1 (np.ndarray): array 1
        x2 (np.ndarray): array 2

    Returns:
        float: euclidean difference
    """
    return np.sqrt(np.square(x1 - x2).sum())
