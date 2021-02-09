import numpy as np

from src.base import SupervisedModel


class DecisionTreeClassifier(SupervisedModel):
    
    @staticmethod
    def _entropy(y: np.ndarray) -> float:
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

    @staticmethod
    def _information_gain(arr: np.ndarray, y: np.ndarray) -> float:
        """Popular information theoretic approach for selecting features in decision trees, based on entropy.

        Args:
            arr (np.ndarray): array of labels
            y (np.ndarray): array of classes

        Returns:
            float: information gain
        """
        n = len(arr)

        orig_entropy = DecisionTreeClassifier._entropy(y)
        after_split_entropy = 0.0

        # get after split entropy for each label
        for label, count in zip(np.unique(arr, return_counts=True)):
            after_split_entropy += (count/n)*DecisionTreeClassifier._entropy(y[arr==label])

        return orig_entropy - after_split_entropy
