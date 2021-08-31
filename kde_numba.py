import numpy as np
import numba as nb


@nb.jit(nopython=True)
def KernelDensityEstimation_numba(xrange, samples):
    xrange = np.linspace(xrange[0], xrange[-1], 1000)
    sample = samples
    N = samples.shape[0]
    h = ((4 * (np.std(samples) ** 5)) / (3 * len(samples))) ** (1/5)
    interval = xrange[1] - xrange[0]

    estimation = np.zeros(xrange.shape[0])

    for i in range(N):
        estimation += np.exp(-(((xrange - sample[i]) / h) ** 2 / 2)) / h

    estimation /= np.sum(estimation) * interval

    return estimation

