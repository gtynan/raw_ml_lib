import numpy as np
from scipy.stats import mode

from src.base import SupervisedModel
from src.distance_functions import hamming_distance, absolute_difference, euclidean_distance

distance_functions = {
    "hamming": hamming_distance,
    "absolute": absolute_difference,
    "euclidean": euclidean_distance
}


class KNNClassifier(SupervisedModel):
    
    def __init__(self, k:int) -> None:
        """KNN model

        Args:
            k (int): # of nearest neighbours to consider
        """
        self.k = k

    def predict(self, X: np.ndarray, func: str = "euclidean") -> np.ndarray:
        """Predict class labels for X data

        Args:
            X (np.ndarray): data you wish to predict class values for
            func (str): distance function to apply options = [hamming, absolute, euclidean]

        Returns:
            np.ndarray: predictions
        """
        # save memory by initialisng our array size
        res = np.empty(shape=(len(X), ))

        # TODO vectorise
        for i, row in enumerate(X):
            # distance functions compare 1:1 thus must apply along axis
            distances = np.apply_along_axis(func1d=distance_functions[func],
                                            axis=1, 
                                            arr=self.X,
                                            x2=row)
            # indicies of k closest rows
            k_min_indicies = np.argpartition(distances, self.k)[:self.k]
            # classify with most common label
            res[i] = mode(self.y[k_min_indicies]).mode
        
        return res
