import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier as SKDecisionTreeClassifier

from src.supervised_models.decision_tree import DecisionTreeClassifier

class TestDecisionTreeClassifier:

    @pytest.fixture(scope="function")
    def model(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier()

    def test_fit(self, model, dummy_cat_X, dummy_cat_y):
        model.fit(dummy_cat_X, dummy_cat_y)
        # ensure decision tree created
        assert isinstance(model.tree, dict)

    def test_predict(self, model, dummy_cat_X, dummy_cat_y):
        y = SKDecisionTreeClassifier(criterion="entropy")\
                                        .fit(dummy_cat_X, dummy_cat_y)\
                                        .predict(dummy_cat_X)

        model.fit(dummy_cat_X, dummy_cat_y)
        y_hat = model.predict(dummy_cat_X) 
        np.testing.assert_array_equal(y, y_hat)
