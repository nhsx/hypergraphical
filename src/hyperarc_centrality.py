import numpy as np


# @numba.njit(
#     nogil=True,
#     fastmath=True,
# )
def iterate_eigencentrality_vector(incidence_matrix, weight, vector):
    """
    This function performs one Chebyshev iteration to calculate the
    largest eigenvalue and corresponding eigenvector of the either the
    standard or dual hypergraph depending on the orientation of the
    incidence matrix.

    This function calculates M * W * M^T * V whilst setting all the
    diagonal elements of M * W * M^T to zero.

    INPUTS:
    ----------
        incidence_matrix (np.array, dtype=numpy.uint8) : The incidence matrix

        weight (np.array, dtype=np.float64) : A vector of weights, which must
        have the same number of elements as the second axis of matrix.

        vector (np.array, dtype=np.float64) : The vector to multiply the matrix
        by. Must have the same number of elements as the second axis of matrix.

    RETURNS:
    ----------
        result (np.array, dtype=np.float64) : The result of
        matrix * weight * transpose(matrix) * vector with diagonal elements of
        matrix * weight * transpose(matrix) set to zero.
    """

    # we are calculating [M W M^T - diag(M W M^T)] v
    term_1 = np.zeros_like(vector)

    # 1) W M^T
    weighted_incidence = np.zeros_like(incidence_matrix, dtype=np.float64)
    for i in range(weighted_incidence.shape[0]):
        for j in range(weighted_incidence.shape[1]):
            weighted_incidence[i, j] += incidence_matrix[i, j] * weight[j]

    # 2) W M^T v
    intermediate = np.zeros(weighted_incidence.shape[1], dtype=np.float64)
    for k in range(weighted_incidence.shape[1]):
        for j in range(len(vector)):
            intermediate[k] += weighted_incidence[j, k] * vector[j]

    # 3) M W M^T v
    for i in range(len(vector)):
        for k in range(weighted_incidence.shape[1]):
            term_1[i] += incidence_matrix[i, k] * intermediate[k]

    # 4) diag(M W M^T v) can be done in one step using W M^T from before
    subt = np.zeros_like(vector)
    for i in range(len(vector)):
        for k in range(weighted_incidence.shape[1]):
            subt[i] += (
                incidence_matrix[i, k] * weighted_incidence[i, k] * vector[i]
            )  # noqa: E501

    # 5) subtract one from the other.
    result = np.zeros_like(vector)
    for i in range(len(vector)):
        result[i] = term_1[i] - subt[i]

    return result


# @numba.njit(fastmath=True, nogil=True)
def matrix_mult(matrix, vector):
    """
    For matrices whose long axis is very large, compute matrix multiplication
    of matrix and vector (where the vector is actually a diagonal matrix).

    INPUTS:
    -------------------
        matrix (np.array, dtype=np.float64) : Matrix whose first axis is very
        large.

        vector (np.array, dtype=numpy.float64) : The vector to multiply the
        matrix by. Must have the same number of elements as the second axis of
        matrix.
    """
    # Compute matrix multiplication of a vector and matrix
    nrows, ncols = matrix.shape
    result = np.zeros_like(matrix, dtype=np.float64)
    for i in range(nrows):
        for j in range(ncols):
            result[i, j] += vector[i] * matrix[i, j]

    return result


def eigenvector_centrality(
    incidence_matrix,
    edge_weight,
    node_weight,
    rep="standard",
    tolerance=1e-6,
    max_iterations=100,
    random_seed=12345,
    weight_resultant=False,
    return_error=False,
):
    """
    This function uses a Chebyshev algorithm to find the eigenvector
    and largest eigenvalue (corresponding to the eigenvector
    centrality) from the adjacency matrix calculated using the incidence
    matrix, a set of weights (for either edges/weights) and another set
    of weights (for either nodes/edges) for the potential resultant weighting.

    Depending on the representation chosen, the length of the returned
    eigenvector will be the size of the first dimension of the tail/head
    incidence matrix, i.e. for standard it will be the size of nodes and
    dual it will be the size of the hyperarcs.

    One may use this function to compute the centrality of the standard or
    the dual hypergraph of the undirected representation of the directed
    hypergraph.

    INPUTS
    --------------------
        incidence_matrix (np.array, dtype=np.uint8) : The incidence matrix.

        edge_weight (np.array, dtype=np.float64) : A vector of edge weights,
        which must have the same number of elements as the second axis of
        matrix.

        node_weight (np.array, dtype=np.float64) : A vector of node weights,
        which must have the same number of elements as the first axis of
        matrix.

        rep (string, optional) : The representation of the hypergraph for which
        to calculate the eigenvector centrality. Options are "standard" or
        "dual". (default: "standard")

        tolerance (float, optional) :The difference between iterations in the
        eigenvalue at which to assume the algorithm has converged
        (default: 1e-6).

        max_iterations (int, optional) : The maximum number of iterations (of
        the power method) to perform before terminating the algorithm and
        assuming it has not converged (default: 100)

        weight_resultant (bool) : Flag to tell function to include the node
        weights during computation of the transition matrix for the dual
        directed hypergraph.

        random_seed (int, optional) : The random seed to use in generating the
        initial vector (default: 12345)

        return_error (bool) : Flag to return eigenvector, but also eigenvalue,
        its error and the iteration of convergence.
    """
    # 1) setup up hypergraph type and initialise eigenvector
    rng = np.random.default_rng(random_seed)
    if rep == "standard":
        res_weight = node_weight
        weight = edge_weight
        inc_mat_original = incidence_matrix
        nrows, ncols = inc_mat_original.shape
    else:
        res_weight = edge_weight
        weight = node_weight
        inc_mat_original = incidence_matrix.T
        nrows, ncols = inc_mat_original.shape
    old_eigenvector_estimate = rng.random(nrows, dtype="float32")

    # 2) Normalise using L_2 norm and apply weighted resultant
    norm_factor = np.linalg.norm(old_eigenvector_estimate)
    old_eigenvector_estimate /= norm_factor
    eigenvalue_estimates, eigenvalue_error_estimates = [], []
    if weight_resultant:
        res_weight_sqroot = np.sqrt(res_weight)
        inc_mat = matrix_mult(inc_mat_original, res_weight_sqroot)
    else:
        inc_mat = inc_mat_original

    # 3) do the Chebyshev
    # In principle, the body of this loop could be compiled with Numba.
    # However, _iterate_vector() is quadratic in long_axis_size, whereas
    # all the other operations here are linear in it, so we are spend very
    # little time in the rest of this loop body.
    for iteration in range(max_iterations):
        print("\rRunning iteration {}...".format(iteration), end="")

        # Iterate over vector
        new_eigenvector_estimate = iterate_eigencentrality_vector(
            inc_mat, weight, old_eigenvector_estimate
        )

        # To estimate eigenvalue and error, take ratio of new to old
        # eigenvector ignoring zeros, whose mean and standard deviation
        # represent new estimate and error
        mask = (new_eigenvector_estimate != 0) & (
            old_eigenvector_estimate != 0
        )  # noqa: E501
        iter_eigenvalue_estimates = (
            new_eigenvector_estimate[mask] / old_eigenvector_estimate[mask]
        )
        eigenvalue_estimate = iter_eigenvalue_estimates.mean()
        eigenvalue_error_estimate = iter_eigenvalue_estimates.std()

        # Append new estimates for eigenvalue and error
        eigenvalue_estimates.append(eigenvalue_estimate)
        eigenvalue_error_estimates.append(eigenvalue_error_estimate)

        # Check to see if tolerance has been met
        if eigenvalue_error_estimate / eigenvalue_estimate < tolerance:
            eig_est = float(f"{eigenvalue_estimate:.3g}")
            eig_err = float(f"{eigenvalue_error_estimate:.3f}")
            print(
                (
                    f"\nConverged at largest eigenvalue {eig_est:g} "
                    f"± {eig_err:f} after {iteration} iterations"
                )
            )
            break

        # Normalise to maintain probability space and prevent overflows
        norm_factor = np.linalg.norm(new_eigenvector_estimate)
        old_eigenvector_estimate = np.array(
            new_eigenvector_estimate / norm_factor
        )  # noqa: E501

    # Break out of algorithm if maximum iterations were reached
    else:
        print(f"\nFailed to converge after {iteration} iterations.")
        eig_est = eigenvalue_estimate
        eig_err = eigenvalue_error_estimate
        print(
            (
                f"Last estimate was {float(f'{eig_est:.3g}'):g} "
                f"± {float(f'{eig_err:.3f}'):f}"
            )
        )

    # Once out of iterative eigenvalue estimator, define eigenvalue,
    # normalise eigenvector and return
    eigenvalue = eigenvalue_estimate
    norm_factor = np.linalg.norm(new_eigenvector_estimate)
    eigenvector = np.array(new_eigenvector_estimate / norm_factor)

    # Depending on if outputing error
    if return_error:
        output = (
            eigenvector,
            (eigenvalue, eigenvalue_error_estimate),
            iteration,
        )
    else:
        output = eigenvector

    return output
