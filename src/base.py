from abc import ABC, abstractmethod
import numpy as np

class AbstractBaseModel(ABC):
    """All models both supervised and unsupervised will inherit this class
    """

    @property
    @abstractmethod
    def X(self) -> np.ndarray:
        pass

    @X.setter
    @abstractmethod
    def X(self, value) -> None:
        pass

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class BaseModel(AbstractBaseModel):
    """All models both supervised and unsupervised will inherit this class
    """

    @property
    def X(self) -> np.ndarray:
        return self._X

    @X.setter
    def X(self, value: np.ndarray) -> None:
        # must be a numpy array
        assert isinstance(value, np.ndarray)
        self._X = value

    def fit(self, X: np.ndarray) -> None:
        self.X = X

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SupervisedModel(BaseModel):
    """All supervised models will inherit this class
    """
    
    @property
    def y(self) -> np.ndarray:
        return self._y

    @y.setter
    def y(self, value) -> None:
        assert isinstance(value, np.ndarray)
        # ensure flat array with same length as X
        assert value.ndim == 1
        assert len(value) == len(self.X)
        self._y = value
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        super().fit(X)
        self.y = y
