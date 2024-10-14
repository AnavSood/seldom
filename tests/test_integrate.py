import numpy as np
import pytest
import scipy
import seldom as sd
import warnings


def _selection_func(x, z, gamma):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return scipy.integrate.quad(
            lambda u: np.prod(np.clip(1 - (gamma / (1-gamma) * (x-z) + u), 0, 1)),
            a=0,
            b=1,
        )[0]


@pytest.mark.parametrize("n", [1, 10, 20])
@pytest.mark.parametrize("gamma", [0.1, 0.5, 0.9])
def test_selection_func(n, gamma, seed=0):
    np.random.seed(seed)
    ps = np.sort(np.random.uniform(0, 1, n))
    x = ps[0]
    z = np.array(ps[1:], copy=True)
    actual = sd.integrate.selection_func(x, z, gamma)
    expected = _selection_func(x, z, gamma)

    assert np.allclose(actual, expected, atol=1e-4)


@pytest.mark.parametrize("n", [1, 10, 20])
@pytest.mark.parametrize("gamma", [0.1, 0.5, 0.9])
def test_integrate_selection_func(n, gamma, seed=0):
    def _integrate_selection_func(p, z, gamma):
        return scipy.integrate.quad(
            lambda x: _selection_func(x, z, gamma),
            a=0,
            b=p,
        )[0]

    np.random.seed(seed)
    ps = np.sort(np.random.uniform(0, 1, n))
    p = ps[0]
    z = np.array(ps[1:], copy=True)
    actual = sd.integrate.integrate_selection_func(p, z, gamma)
    expected = _integrate_selection_func(p, z, gamma)

    assert np.allclose(actual, expected, atol=1e-4)


@pytest.mark.parametrize("seed", [0, 5, 10, 20])
def test_gauss_fd(seed):
    def _gauss_fd(a, t):
        return (
            scipy.stats.norm.expect(
                lambda x: scipy.stats.norm.sf(t - x),
                lb=a,
            ) / scipy.stats.norm.sf(t / np.sqrt(2))
        )

    np.random.seed(seed)
    a = np.random.normal(0, 1)
    t = np.random.normal(0, 1)
    actual = sd.integrate.gauss_fd(a, t)
    expected = _gauss_fd(a, t)

    assert np.allclose(actual, expected, atol=1e-4)