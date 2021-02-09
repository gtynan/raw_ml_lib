import pytest
import numpy as np

from src.supervised_models.decision_tree import DecisionTreeClassifier

class TestDecisionTreeClassifier:

    @pytest.fixture(scope='function')
    def model(self):
        return DecisionTreeClassifier()

    @pytest.mark.parametrize('arr, res', [([1, 1, 1], 0), ([1, 1, 1, 0, 0, 0], 1)])
    def test_entropy(self, arr, res):
        assert DecisionTreeClassifier._entropy(arr) == res


    @pytest.mark.parametrize('arr, y, res', [
        (np.array([0,0,1,1,1,1,2,2,2,2,2,2]), np.array([0,0,1,1,1,1,1,1,0,0,0,0]), 0.541),
        (np.array([0,0,1,1,1,1,2,2,3,3,3,3]), np.array([0,1,0,1,0,1,0,1,0,1,0,1]), 0),
    ])
    def test_information_gain(self, arr, y, res):
        pytest.approx(DecisionTreeClassifier._information_gain(arr, y), res)