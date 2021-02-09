import pytest
import numpy as np

random_settings = np.random.default_rng(seed=2)

@pytest.fixture(scope='session')
def dummy_X() -> np.ndarray:
    return random_settings.random((10, 5))


@pytest.fixture(scope='session')
def dummy_y() -> np.ndarray:
    return random_settings.random((10, ))


@pytest.fixture(scope='session')
def dummy_cat_y() -> np.ndarray:
    # random number 0, 1 or 2
    return np.random.randint(3, size=(10, ))
