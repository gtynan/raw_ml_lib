from numpy.testing import assert_array_equal
import pytest
import numpy as np
from sklearn.naive_bayes import CategoricalNB

from src.supervised_models.naive_bayes import NaiveBayesClassifier


class TestNaiveBayesClassifier:

    @pytest.fixture(scope='function')
    def model(self) -> NaiveBayesClassifier:
        return NaiveBayesClassifier()

    def test_fit(self, model, dummy_cat_X, dummy_cat_y):
        model.fit(dummy_cat_X, dummy_cat_y)
        assert isinstance(model.class_probs, np.ndarray)
        assert isinstance(model.naive_probs, dict)

    def test_predict(self, model, dummy_cat_X, dummy_cat_y):
        # reduce alpha to ensure no smoothing
        y = CategoricalNB(alpha=1.0e-10)\
                            .fit(dummy_cat_X, dummy_cat_y)\
                            .predict(dummy_cat_X)

        model.fit(dummy_cat_X, dummy_cat_y)
        y_hat = model.predict(dummy_cat_X)

        np.testing.assert_array_equal(y, y_hat)
