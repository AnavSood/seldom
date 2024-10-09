from . import seldom_core as core
from typing import Tuple, Union
import numpy as np
import scipy


def selection_func(
    x: float,
    z: np.ndarray,
    gamma: float,
    *,
    quadrature: Union[Tuple[np.ndarray, np.ndarray], None] =None,
):
    """Computes the selection function.

    The selection function is defined by

    .. math::
        \\begin{align*}
            s(x; z, \\gamma)
            :=
            \\int_0^1
            \\max\\left(
                \\min\\left(
                    \\prod\\limits_{j=1}^m \\left(
                        1 - \\left(
                            \\frac{\\gamma}{1-\\gamma} (x - z_j) + u
                        \\right)
                    \\right), 
                    1
                \\right),
                0
            \\right)
            du
        \\end{align*}

    Parameters:
    -----------
    x : float
        The first p-value.
    z : ndarray
        The other p-values.
    gamma : float
        Degree of randomness.
    quadrature : Union[Tuple[ndarray, ndarray], None], optional
        The quadrature points and weights as given by
        :func:`scipy.special.roots_legendre`.
        If ``None``, it is generated internally.
        Default is ``None``.

    Returns:
    -------- 
    s : float
        Selection function :math:`s(x, z)`.
    """
    if quadrature is None:
        quadrature = scipy.special.roots_legendre(100)
    return core.integrate.selection_func(x, z, gamma, *quadrature)


def integrate_selection_func(
    p: float,
    z: np.ndarray,
    gamma: float,
    *,
    quadrature: Union[Tuple[np.ndarray, np.ndarray], None] =None,
):
    """Computes the integral of the selection function.

    The integral of the selection function is given by
    
    .. math::
        \\begin{align*}
            S(p; z, \\gamma)
            :=
            \\int_0^p s(x; z, \\gamma) dx
        \\end{align*}

    Parameters: 
    -----------
    p : float
        Integration upper bound.
    z : ndarray
        The other p-values.
    gamma : float
        Degree of randomness.
    quadrature : Union[Tuple[ndarray, ndarray], None], optional
        The quadrature points and weights as given by
        :func:`scipy.special.roots_legendre`.
        If ``None``, it is generated internally.
        Default is ``None``.

    Returns:
    --------
    S : float
        Integral of the selection function :math:`S(p; z, \\gamma)`.
    """
    if quadrature is None:
        quadrature = scipy.special.roots_legendre(100)
    return core.integrate.integrate_selection_func(p, z, gamma, *quadrature)