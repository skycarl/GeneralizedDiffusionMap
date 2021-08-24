"""Target measure diffusion map."""

import numpy as np
from GDFM import utils


def TMDMap(data, target_distribution, eps=1.0, beta=1.0, n_dim=2):
    """Computes the Target Measure Diffusion Map (TMDMap) from Banisch et al., (2017).

    Parameters
    ----------
    data : np.array
        (N x dim) array of N data points of dimension dim
    target_distribution : np.array
        Target distribution; len(target_distribution) must equal len(data)
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

    if len(target_distribution) != len(data):
        raise ValueError('len(target_distribution) must equal len(data)')

    a, q = utils.kernel(data, eps)

    # Compute weights
    weights = np.sqrt(target_distribution)
    d = np.diag(weights*q)

    # Compute the K_{\epsilon, \pi} matrix
    k_eps = np.matmul(a, d)
    k_eps = np.matmul(np.diag(1/np.sum(k_eps, 1)), k_eps)

    # Compute the L_{\epsilon, \pi} matrix
    l_loc = 4/(beta*eps) * (k_eps - np.diag(np.sum(k_eps, 1)))

    # Compute top eigenvalues and eigenvectors
    eig_val, eig_vec = utils.top_eigs(l_loc, n_dim)

    return eig_val, eig_vec
