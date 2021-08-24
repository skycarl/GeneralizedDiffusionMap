"""Utilities for diffusion maps."""

import numpy as np
from scipy.spatial import cKDTree


def kernel(data, eps=0.01):
    """Kernel for kernel density estimate

    Parameters
    ----------
    data : np.array
        (N x dim) array of N data points of dimension dim
    eps : float, optional
        scaling parameter for the kernel, by default 0.01

    Returns
    -------
    tuple
        (A, q) - kernel A and the density q
    """

    r = np.sqrt(2*eps)

    # Find all points and distances within r
    tree = cKDTree(data)
    d = tree.sparse_distance_matrix(tree, max_distance=4*r).toarray()

    # Isotropic kernel; see equation (1)
    k_eps = np.exp(-(d**2)/eps)

    # We want 0 where there was an invalid distance and 1 along diagonal
    k_eps[d == 0] = 0
    np.fill_diagonal(k_eps, 1)

    # q is (kernel density estimate)^{-1}
    q = 1/np.sum(k_eps, axis=1)

    return k_eps, q


def top_eigs(mat, n_eigs):
    """Find the top `n_eigs` eigenvalues and eigenvectors from the input matrix `mat`.

    Parameters
    ----------
    mat : np.array
        Matrix from which to compute eigenvalues and eigenvectors
    n_eigs : int
        Number of top eigenvectors and eigenvalues to return

    Returns
    -------
    tuple
        (eigenvalues, eigenvectors)
    """

    # Compute eigenvalues and eigenvectors
    eig_val, eig_vec = np.linalg.eig(mat)

    # Get the top n_eigs eigenvalues
    sort_idx = np.argsort(eig_val)[::-1][0:n_eigs]
    eig_val.sort()
    eig_val = eig_val[::-1]
    eig_val = eig_val[0:n_eigs]

    # Get the real parts of the top red_dim eigenvectors
    eig_vec = np.real(eig_vec[sort_idx].T)

    return eig_val, eig_vec
