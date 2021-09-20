"""
Try to synthesize a random unitary matrix
"""

from qcontrol_polyopt import QControlPolyOpt
import numpy as np
import scipy as sp
from scipy import linalg
from multiprocessing import Pool
from qutip import propagator, Qobj

def haar_measure(n):
    """
    A random unitary matrix distributed with Haar measure.

    The algorithm is taken from https://arxiv.org/abs/math-ph/0609050
    :param n: (int) dimension
    :return: numpy.array
    """
    z = (sp.randn(n, n) + 1j * sp.randn(n, n)) / sp.sqrt(2.0)
    q,r = linalg.qr(z)
    d = sp.diagonal(r)
    ph = d / sp.absolute(d)
    q = sp.multiply(q, ph, q)
    return q


def qcontrol_to_get_rand_unit(args):
    """

    :param args: control, H0, V, seed
    :return: (np.float) the distance norm between the target and synthesized unitary
    """
    control, H0, V, seed = args

    # Set the seed for random number generation to avoid the artifact described in
    #   https://stackoverflow.com/questions/24345637/why-doesnt-numpy-random-and-multiprocessing-play-nice
    # It is recommended that seeds be generate via the function get_seeds (see below)
    np.random.seed(seed)

    # get a random unitary as a target gate to be synthesized
    U_target = haar_measure(H0.shape[0])

    # find the control pulse via polynomial optimization
    control.get_controls(U_target)

    if control.u_opt is None:
        # the optimization has not converged
        return np.nan
    else:
        # Obtain the unitary that the found control synthesized
        U_opt = propagator(
            [H0, [V, str(control.u_opt(control.t))]], control.T,
        )
        # return the Frobenius norm between the unitaries
        return (U_target - U_opt).norm(norm='fro')


def get_seeds(size):
    """
    Generate unique random seeds for subsequently seeding them into random number generators in multiprocessing simulations
    This utility is to avoid the following artifact:
        https://stackoverflow.com/questions/24345637/why-doesnt-numpy-random-and-multiprocessing-play-nice
    :param size: number of samples to generate
    :return: numpy.array of np.uint32
    """
    # Note that np.random.seed accepts 32 bit unsigned integers

    # get the maximum value of np.uint32 can take
    max_val = np.iinfo(np.uint32).max

    # A set of unique and random np.uint32
    seeds = set()

    # generate random numbers until we have sufficiently many nonrepeating numbers
    while len(seeds) < size:
        seeds.update(
            np.random.randint(max_val, size=size, dtype=np.uint32)
        )

    # make sure we do not return more numbers that we are asked for
    return np.fromiter(seeds, np.uint32, size)


if __name__ =='__main__':
    import matplotlib.pyplot as plt

    # Quantum system is taken from https://github.com/q-optimize/c3/blob/master/examples/two_qubits.ipynb

    # Drift Hamiltonian
    H0 = np.array([[0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j],
                   [0.00000000e+00 + 0.j, 3.21505101e+10 + 0.j, 0.00000000e+00 + 0.j],
                   [0.00000000e+00 + 0.j, 0.00000000e+00 + 0.j, 6.23173079e+10 + 0.j]])

    # Normalize the drift Hamiltonian by its maximum.
    # This normalization factor can be accounted for by rescaling the time.
    H0 /= H0.max()

    # Control Hamiltonian
    V = np.array([[0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [1. + 0.j, 0. + 0.j, 1.41421356 + 0.j],
                  [0. + 0.j, 1.41421356 + 0.j, 0. + 0.j]])

    # Normalize the control Hamiltonian by its maximum.
    # This normalization factor can be accounted for by rescaling the control field.
    V /= V.max()

    #  Initialize class for polynomial control optimization
    control = QControlPolyOpt(H0=H0, V=V)

    H0 = Qobj(H0)
    V = Qobj(V)

    # the iterator to launch (2 * chunksize) trajectories
    chunksize = 1
    iter_arg = ((control, H0, V, seed) for seed in get_seeds(2 * chunksize))

    # run tests on multiple cores
    with Pool() as pool:
        results = list(pool.imap_unordered(qcontrol_to_get_rand_unit, iter_arg, chunksize=chunksize))

    print(results)
    # results = np.ravel(results)
