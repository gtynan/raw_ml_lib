import numpy as np


def entropy(y: np.ndarray) -> float:
    """Entropy provides a measure of impurity

    Args:
        y (np.ndarray): array of classes

    Returns:
        float: entropy
    """
    n = len(y)

    # count occurances of each unique label
    _, label_counts = np.unique(y, return_counts=True)
    # relative frequency of each unique label
    p = label_counts / n

    return -(p * np.log2(p)).sum()


def information_gain(arr: np.ndarray, y: np.ndarray) -> float:
    """Popular information theoretic approach for selecting features in decision trees, based on entropy.

    Args:
        arr (np.ndarray): array of labels
        y (np.ndarray): array of classes

    Returns:
        float: information gain
    """
    n = len(arr)

    orig_entropy = entropy(y)
    after_split_entropy = 0.0

    labels, counts = np.unique(arr, return_counts=True)
    # get after split entropy for each label
    for label, count in zip(labels, counts):
        after_split_entropy += (count/n)*entropy(y[arr==label])

    return orig_entropy - after_split_entropy


def get_max_information_gain_col(X: np.ndarray, y: np.ndarray) -> int:
    """Given a matrix of values find column with max information gain

    Args:
        X (np.ndarray): independent variables
        y (np.ndarray): dependent variable

    Returns:
        int: independent variable column with max information gain
    """
    col_info_gain = np.apply_along_axis(func1d=information_gain,
                                        axis=0, arr=X, y=y)
    return np.argmax(col_info_gain)
