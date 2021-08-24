"""Local kernel diffusion map."""

import numpy as np
from GDFM import utils


def LKDMap(data, b, eps=1.0, beta=1.0, n_dim=2):
    """Computes the Local Kernel Diffusion Map (LKDMap) from Banisch et al., (2017).

    Parameters
    ----------
    data : np.array
        (N x dim) array of N data points of dimension dim
    b : np.array
        Array of velocities; len(b) must equal len(data)
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

    if len(b) != len(data):
        raise ValueError('len(b) must equal len(data)')

    # Construct K_{\epsilon} and K_{\epsilon}^{A, b}
    _, q = utils.kernel(data, eps)
    w_dist, _ = utils.kernel(data-eps*b, eps)
    w = np.exp(-beta*w_dist**2 / (4*eps))

    # Form diagonal matrix D_eps
    d_q = np.diag(q)

    # Diagonalize W
    w = np.matmul(w, d_q)

    # Build the LKDMap matrices
    l_loc = np.matmul(np.diag(1/np.sum(w, 1)), w)
    l_loc = 1/eps*(l_loc - np.diag(np.sum(l_loc, 1)))

    # Compute top eigenvalues and eigenvectors
    eig_val, eig_vec = utils.top_eigs(l_loc, n_dim)

    return eig_val, eig_vec
