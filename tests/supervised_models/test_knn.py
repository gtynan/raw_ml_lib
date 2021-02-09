import pytest
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from src.supervised_models.knn import KNNClassifier

class TestKNNClassifier:

    @pytest.fixture(scope="function")
    def model(self) -> KNNClassifier:
        return KNNClassifier(k=3)

    def test_predict(self, model, dummy_X, dummy_cat_y): 
        # compare our predictions with sklearn library
        y = KNeighborsClassifier(n_neighbors=3).fit(X=dummy_X, y=dummy_cat_y).predict(dummy_X)
   
        model.fit(X=dummy_X, y=dummy_cat_y)
        y_hat = model.predict(dummy_X)

        np.testing.assert_array_equal(y, y_hat)
