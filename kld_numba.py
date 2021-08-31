from kde_numba import *
import numpy as np


@nb.jit(nopython=True)
def entropy_nb(p, q):

    p /= np.sum(p)
    q /= np.sum(q)

    rel_ent = 0
    for i in range(p.shape[0]):
        rel_ent += p[i] * np.log(p[i] / q[i])

    return rel_ent


def kl_divergence_numba(p, q, data_range):
    pdf_p = KernelDensityEstimation_numba(data_range, p) + 1e-12
    pdf_q = KernelDensityEstimation_numba(data_range, q) + 1e-12
    return entropy_nb(pdf_p, pdf_q)