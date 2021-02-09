import pytest
import numpy as np

from src.base import AbstractBaseModel, BaseModel, SupervisedModel

class TestAbstractBaseModel:

    def test_X(self):
        # test getter
        with pytest.raises(TypeError):
            AbstractBaseModel().X = 1
        # test setter
        with pytest.raises(TypeError):
            AbstractBaseModel().X
        
    def test_fit(self):
        with pytest.raises(TypeError):
            AbstractBaseModel().fit([])

    def test_predict(self):
        with pytest.raises(TypeError):
            AbstractBaseModel().predict([])

class TestBaseModel:

    @pytest.fixture(scope='function')
    def model(self) -> BaseModel:
        return BaseModel()

    def test_X(self, model, dummy_X):
        # error raised if X not set
        with pytest.raises(AttributeError):
            model.X
        # test setter
        model.X = dummy_X
        # test getter
        np.testing.assert_array_equal(model.X, dummy_X)
    
    def test_fit(self, model, dummy_X):
        model.fit(dummy_X)
        # we expect fit to set our X property
        np.testing.assert_array_equal(model.X, dummy_X)

    def test_predict(self, model, dummy_X):
        with pytest.raises(NotImplementedError):
            model.predict(dummy_X)

class TestSupervisedModel:

    @pytest.fixture(scope='function')
    def model(self) -> SupervisedModel:
        return SupervisedModel()

    def test_y(self, model, dummy_X, dummy_y):
        # cannot set y before X
        with pytest.raises(AttributeError):
            model.y = dummy_y
        # set X
        model.X = dummy_X
        # error raised if y not set
        with pytest.raises(AttributeError):
            model.y
        # test setter
        model.y = dummy_y
        # test getter
        np.testing.assert_array_equal(model.y, dummy_y)
        

    def test_fit(self, model, dummy_X, dummy_y):
        model.fit(dummy_X, dummy_y)
        np.testing.assert_array_equal(model.X, dummy_X)
        np.testing.assert_array_equal(model.y, dummy_y)
