"""Typical diffusion map."""

import numpy as np
from GDFM import utils


def diffmap(data, eps=1.0, beta=1.0, n_dim=2):
    """Computes the generalized diffusion map from Coifman and Lafon (2010).

    Parameters
    ----------
    data : np.array
        (N x dim) array of N data points of dimension dim
    eps : float, optional
        Scaling parameter for the kernel, by default 1.0
    beta : float, optional
        Inverse temperature from the sampled Boltzmann distribution, by default 1.0
    red_dim : int, optional
        Number of dimensions to reduce to, by default 2

    Returns
    -------
    tuple
        (eig_val, eig_vec); first n_dim eigenvalues and eigenvectors of L
    """

    a, q = utils.kernel(data)

    # Diagonal matrix of kernel density estimate
    d = np.diag(np.sqrt(q))

    # Normalize matrix A
    a_norm = np.matmul(np.matmul(d, a), a)

    k = np.matmul(np.diag(1/np.sum(a_norm, axis=1)), a_norm)
    l_diff = 4/(beta*eps)*(k - np.diag(np.sum(k, axis=1)))

    # Compute top eigenvalues and eigenvectors
    eig_val, eig_vec = utils.top_eigs(l_diff, n_dim)

    return eig_val, eig_vec
