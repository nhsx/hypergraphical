import pytest
import os
import math

os.chdir("C:/Users/ZOEHANCOX/OneDrive - NHS England/hypergraphical")
from src import numpy_utils, utils


# def test_n_choose_k():
#     """given some values for n and k ensure the numpy_utils.N_choose_k
#     equals n! / (k! * (n - k)!).
#     Both utils.N_choose_k (Numba version) and numpy_utils.N_choose_k
#     are compared to the value calculated by hand."""

#     n = 3
#     k = 2
#     exp_ans = 3
#     numpy_ans = numpy_utils.N_choose_k(n, k)
#     numba_ans = utils.N_choose_k(n, k)

#     assert numpy_ans == exp_ans
#     assert numba_ans == exp_ans

#     n = 8
#     k = 5
#     exp_ans = 56
#     numpy_ans = numpy_utils.N_choose_k(n, k)
#     numba_ans = utils.N_choose_k(n, k)
#     tol = 1e-9

#     assert math.isclose(numpy_ans, exp_ans, rel_tol=tol, abs_tol=tol)
#     assert math.isclose(numba_ans, exp_ans, rel_tol=tol, abs_tol=tol)


def test_max_b_hyperarcs():
    """Testing the N_max_hyperarcs function to ensure it calculates
    the maximum number of B-hyperarcs correctly.

    This includes self-edges, as it's looking at directed hypergraphs.
    """

    n_dis = 3
    numpy_ans = numpy_utils.N_max_hyperarcs(n_dis, b_hyp=True)
    exp_ans = 12
    assert numpy_ans == exp_ans

    n_dis = 2
    numpy_ans = numpy_utils.N_max_hyperarcs(n_dis, b_hyp=True)
    exp_ans = 4
    assert numpy_ans == exp_ans


def test_max_hyperedges():
    """Testing the N_max_hyperedges function to ensure it calculates
    the maximum number of hyperedges correctly.

    This does not include self-edges, as it's looking at undirected
    hypergraphs.
    """
    n_dis = 3
    numpy_ans = numpy_utils.N_max_hyperedges(n_dis)
    exp_ans = 4
    assert numpy_ans == exp_ans

    n_dis = 4
    numpy_ans = numpy_utils.N_max_hyperedges(n_dis)
    exp_ans = 11
    assert numpy_ans == exp_ans


def test_max_bf_hyperarcs():
    """Testing the calculation of the maximum number of BF-hyperarcs is
    correct.

    Hand calculated answer versus numpy calculated.
    """
    n_dis = 3
    exp_ans = 15
    numpy_ans = numpy_utils.N_max_hyperarcs(n_dis, b_hyp=False)
    assert numpy_ans == exp_ans
