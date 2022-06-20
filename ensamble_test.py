"""
Try to synthesize a random reachable unitary matrix
"""

from qcontrol_polyopt import *
import numpy as np
from multiprocessing import Pool
from qutip import propagator, Qobj


def qcontrol_to_get_rand_unit(args):
    """
    :param args: control, H0, V, spd_relax, seed
    :return: U_target, x_exact, norm_diff, obj_poly_val: norm_diff is the distance norm between the target (U_target) and synthesized unitary
    """
    control, H0, V, spd_relax, seed = args

    # Set the seed for random number generation to avoid the artifact described in
    #   https://stackoverflow.com/questions/24345637/why-doesnt-numpy-random-and-multiprocessing-play-nice
    # It is recommended that seeds be generate via the function get_seeds (see below)
    np.random.seed(seed)

    t = control.t

    # Randomly generate the control field
    x_exact_val = np.random.uniform(-1, 1, len(control.x))
    x_test = dict(zip(control.x, x_exact_val))
    u_test = control.u(t).subs(x_test)

    # get a unitary by propagating in the generated random field
    U_target = propagator(
        [H0, [V, str(u_test)]], control.T,
    )

    # find the control pulse via polynomial optimization
    control.get_controls(U_target, spd_relax)

    if control.u_opt is None:
        # the optimization has not converged
        obj_poly_val = norm_diff = np.nan
    else:
        obj_poly_val = control.obj_poly_val

        # Obtain the unitary that the found control synthesized
        U_opt = propagator(
            [H0, [V, str(control.u_opt(t))]], control.T,
        )
        # return the Frobenius norm between the unitaries
        norm_diff = (U_target - U_opt).norm(norm='fro')

    return np.array(U_target, dtype=np.complex), x_exact_val, norm_diff, obj_poly_val


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


if __name__ == '__main__':
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

    np.random.seed(9202021)

    # SPD relaxation level
    spd_relax = 8

    # the iterator to launch (110 * chunksize) trajectories
    chunksize = 1000
    iter_arg = ((control, H0, V, spd_relax, seed) for seed in get_seeds(110 * chunksize))

    # run tests on multiple cores
    with Pool() as pool:
        results = list(pool.imap_unordered(qcontrol_to_get_rand_unit, iter_arg, chunksize=chunksize))

    U_targets, x_exact_vals, norm_diffs, obj_poly_vals = zip(*results)

    np.savez_compressed(
        'ensamble_simulations_spd_relax={}.npz'.format(spd_relax),
        U_targets=U_targets,
        x_exact_vals=x_exact_vals,
        norm_diffs=norm_diffs,
        obj_poly_vals=np.array(obj_poly_vals, dtype=np.float),
    )
