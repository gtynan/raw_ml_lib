from typing import Any, List, Optional
import numpy as np
from scipy.stats import mode

from src.base import SupervisedModel
from src.entropy import get_max_information_gain_col


class DecisionTreeClassifier(SupervisedModel):

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Create decison tree from X and y values

        Args:
            X (np.ndarray): independent variables
            y (np.ndarray): dependent varaible
        """
        super().fit(X, y)
        self.tree = DecisionTreeClassifier._id3(X, y)

    @staticmethod
    def _id3(X: np.ndarray, y: np.ndarray) -> dict:
        """Popular algorithm which repeatedly builds a decision tree from the top down (Quinlan, 1986)

        Args:
            X (np.ndarray): independent variables
            y (np.ndarray): dependent varaible

        Returns:
            dict: decision tree
        """
        col_id = get_max_information_gain_col(X, y)
        # root node
        tree = {col_id: {}}
                
        # unique values in said column
        col_values = np.unique(X[:, col_id])

        for value in col_values:
            # rows where values match specific value
            mask = X[:, col_id] == value
            sub_X, sub_y = X[mask], y[mask]

            y_values, y_counts = np.unique(sub_y, return_counts=True)

            # if only 1 y value associated with current X value define it as the answer
            if len(y_counts) == 1:
                tree[col_id][value] = y_values[0]

            # identical rows with different labels return mode
            elif (sub_X == sub_X[0, :]).all() and ~(sub_y == sub_y[0]).all():
                tree[col_id][value] = mode(y_values).mode[0]

            else:
                tree[col_id][value] = DecisionTreeClassifier._id3(sub_X, sub_y)
        return tree

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict dependent values for X

        Args:
            X (np.ndarray): independent variables

        Returns:
            np.ndarray: prediction
        """
        if X.ndim > 1:
            return np.apply_along_axis(DecisionTreeClassifier._predict,
                                        axis=1,
                                        arr=X,
                                        tree=self.tree).ravel()
        return DecisionTreeClassifier._predict(X, self.tree)

    @staticmethod
    def _predict(X: np.ndarray, tree: dict) -> Optional[Any]:
        """Recursively calls itself to work through tree until answer obtained if one exists
        """
        assert X.ndim == 1

        for col, val in tree.items():
            # x value for associated column
            x_val = X[col]
            if isinstance(val , dict):
                if isinstance(tree[col][x_val], dict):
                    return DecisionTreeClassifier._predict(X, tree[col][x_val])
                return tree[col][x_val]
            return tree[col]
