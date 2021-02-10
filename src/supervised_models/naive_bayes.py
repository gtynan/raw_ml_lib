import numpy as np

from src.base import SupervisedModel


class NaiveBayesClassifier(SupervisedModel):
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Naive Bayes classifier to X and y values

        Args:
            X (np.ndarray): independent variables
            y (np.ndarray): dependent variable
        """
        super().fit(X, y)
        n = len(X)

        self.unique_y, y_counts = np.unique(y, return_counts=True)

        self.class_probs = np.zeros((len(self.unique_y), ))
        self.naive_probs = {}

        # loop through all unique y values to get X naive probs for each
        for y_i, (y_val, y_count) in enumerate(zip(self.unique_y, y_counts)):

            self.class_probs[y_i] = y_count/n
            self.naive_probs[y_val] = {}

            # only consider X rows with associated y value
            sub_X = X[y == y_val]

            # populate naive probs for each X column and each unique value within
            for col_i, col in enumerate(sub_X.T):
                self.naive_probs[y_val][col_i] = {}

                col_vals, col_counts = np.unique(col, return_counts=True)
                for col_val, col_count in zip(col_vals, col_counts):
                    self.naive_probs[y_val][col_i][col_val] = col_count / y_count

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X data

        Args:
            X (np.ndarray): independent variables

        Returns:
            np.ndarray: predictions 
        """
        if X.ndim > 1:
            return np.apply_along_axis(NaiveBayesClassifier._predict,
                                       axis=1,
                                       arr=X,
                                       y_unique=self.unique_y,
                                       class_probs=self.class_probs,
                                       naive_probs=self.naive_probs).ravel()

        return NaiveBayesClassifier._predict(X, np.unique(self.y), self.class_probs, self.naive_probs)

    @staticmethod
    def _predict(X: np.ndarray, y_unique: np.ndarray, class_probs, naive_probs) -> float:
        """"Single X row prediction"""
        # probs for each y value (row) with X (col) odds
        probs = np.zeros((len(y_unique), len(X)))

        for y_i, y_val in enumerate(y_unique):
            for x_i, x_val in enumerate(X):
                try:
                    probs[y_i][x_i] = naive_probs[y_val][x_i][x_val] 
                except:
                    # no prior odds skip as product will = 0
                    break

        probs = probs.prod(axis=1) * class_probs
        
        # if all probs = 0 no prediciton
        if (probs == 0).all():
            return None
        # return y value with highest probs
        return y_unique[np.argmax(probs)]
