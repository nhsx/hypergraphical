import pytest
import os

os.chdir("C:/Users/ZOEHANCOX/OneDrive - NHS England/hypergraphical")
import math
from src import numpy_utils, utils


def test_n_choose_k():
    """given some values for n and k ensure the numpy_utils.N_choose_k
    equals n! / (k! * (n - k)!).
    Both utils.N_choose_k (Numba version) and numpy_utils.N_choose_k
    are compared to the value calculated by hand."""

    n = 3
    k = 2
    exp_ans = 3
    numpy_ans = numpy_utils.N_choose_k(n, k)
    numba_ans = utils.N_choose_k(n, k)

    assert numpy_ans == exp_ans
    assert numba_ans == exp_ans

    n = 8
    k = 5
    exp_ans = 56
    numpy_ans = numpy_utils.N_choose_k(n, k)
    numba_ans = utils.N_choose_k(n, k)
    tol = 1e-9

    assert math.isclose(numpy_ans, exp_ans, rel_tol=tol, abs_tol=tol)
    assert math.isclose(numba_ans, exp_ans, rel_tol=tol, abs_tol=tol)
