"""
Library of classes for quantum control via commutative pop optimization
"""
import numpy as np
from ncpol2sdpa import generate_variables, SdpRelaxation
from scipy.optimize import minimize
from sympy import symbols, lambdify, re, exp, I, integrate, Matrix, Rational

class CABSQCPolyOpt(object):
    """
    Abstract base class for all methods of quantum control via commutative pop optimization
    """
    def __init__(self, *, H0, V, U_target, T, min_module='ncpol2sdpa', npoly=6, **kwargs):
        """
        Constructor
        :param H0: the field free Hamiltonian (use a diagonal matrix for efficiency of symbolic calculations)
        :param V: the interaction matrix describing the coupling with a control field
        :param U_target: the target unitary matrix
        :param T: (flot) the final (terminal) time when we want to reach a target unitary U_target
        :param min_module: (str) Minimization module to be used. Avalible values are ('ncpol2sdpa', 'scipy', and 'both')
        :param npoly: (int) the degree of polynomial approximation for the sought control field
        :param kwargs:
        """
        super(CABSQCPolyOpt, self).__init__(**kwargs)

        # save the data
        self.H0 = H0
        self.V = V
        self.T = T
        self.U_target = Matrix(U_target)
        self.npoly = npoly
        self.min_module = min_module

        # time variable
        t = symbols('t', real=True)

        # the coefficients as varaibles to be minimize over
        self.x = generate_variables('x', self.npoly)

        # define the sought controls symbolically as a polynomial
        self.u = lambdify(t, sum(c * t ** n_ for n_, c in enumerate(self.x)))

    def _get_control_ncpol2sdpa(self):
        """
        Optimize self.obj using ncpol2sdpa via mosek.
        Note self.obj must be defined in a child class.
        :return: None
        """
        t = symbols('t', real=True)

        sdp = SdpRelaxation(self.x)
        sdp.get_relaxation(4, objective=re(self.obj))
        sdp.solve(solver='mosek')

        # extract the values of control
        # if sdp.status == 'optimal':
        opt_vals = [sdp[_] for _ in sdp.variables]

        # save the value of objective function
        self.obj_val = self.obj.subs(
            zip(sdp.variables, (sdp[_] for _ in sdp.variables))
        ).evalf()

        # construct the control
        u_reconstructed = sum(c * t ** n_ for n_, c in enumerate(opt_vals))

        # Covert u_reconstructed to string for qutip
        u_reconstructed_str = str(u_reconstructed)

        # u_reconstructed for plotting
        u_reconstructed = lambdify(t, u_reconstructed, 'numpy')

        return u_reconstructed_str, u_reconstructed

    def _get_control_scipy(self):
        """
        Optimize self.obj using the conjugate gradient in scipy.
        Note self.obj must be defined in a child class.
        :return: None
        """
        t = symbols('t', real=True)

        x = self.x

        # get the objective function, , and hessian
        f = re(self.obj)

        # symbolically get the Jacobian
        jac = [f.diff(_) for _ in x]

        # symbolically get the Hessian
        hess = [[df.diff(_) for _ in x] for df in jac]

        f = lambdify(x, f, 'numpy')
        jac = lambdify(x, jac, 'numpy')
        hess = lambdify(x, hess, 'numpy')

        # Use the conjugate gradient method in scipy since we have both the Jacobian and Hessian of the objective function
        solution = minimize(
            lambda _: f(*_),
            np.zeros(len(x)),
            jac=lambda _: jac(*_),
            hess=lambda _: hess(*_),
            method='CG'
        )

        # save the value of objective function
        self.obj_val = solution.fun

        # construct the control
        u_reconstructed = sum(c * t ** n_ for n_, c in enumerate(solution.x))

        # Covert u_reconstructed to string for qutip
        u_reconstructed_str = str(u_reconstructed)

        # u_reconstructed for plotting
        u_reconstructed = lambdify(t, u_reconstructed, 'numpy')

        return u_reconstructed_str, u_reconstructed

    def get_control(self):
        """
        Find the control via polynomial optimization using the method specified in self.min_module.
        Note self.obj must be defined in a child class.
        :return: None
        """
        if self.min_module == 'ncpol2sdpa':
            self.u_reconstructed = {'ncpol2sdpa': self._get_control_ncpol2sdpa()}

        elif self.min_module == 'scipy':
            self.u_reconstructed = {'scipy', self._get_control_scipy()}

        elif self.min_module == 'both':
            self.u_reconstructed = {'ncpol2sdpa': self._get_control_ncpol2sdpa(), 'scipy': self._get_control_scipy()}

        else:
            raise ValueError("min_module must be one of 'ncpol2sdpa', 'scipy', and 'both'")


class CQCPolyOptDyson(CABSQCPolyOpt):
    """
    Class using the first order Dyson series (with respect to the field interaction) to find the control field
    via commutative pop optimization to reach the taggert unitary.
    """
    def __init__(self, **kwargs):
        """
        Constructor
        :param kwargs: all the arguments are passed to the constructor of class CABSQCPolyOpt
        """
        super(CQCPolyOptDyson, self).__init__(**kwargs)

        ################################################################################################################
        #
        #   Define the objective polynomial function using the Dyson series
        #
        ################################################################################################################

        # aliases
        T = self.T
        u = self.u

        # symbols
        t1, t2, t3 = symbols('t1, t2, t3', real=True)

        # the field free propagator
        U0 = lambda t: exp(-I * self.H0 * t)

        # we assume the initial time is zero
        first_order = -I * integrate(U0(T - t1) * self.V * u(t1) * U0(t1), (t1, 0, T))

        dyson_series = U0(T) + first_order

        self.obj = (
                (dyson_series - self.U_target).norm() ** 2
        ).evalf()

        ################################################################################################################
        #
        #   Find the control via optimization
        #
        ################################################################################################################

        self.get_control()


class CQCPolyOptMagnus(CABSQCPolyOpt):
    """
    Class using the second order Margus series
    (as well as the Lipschitz continuity of a matrix exp to unroll the exponentiation in the Magus formula)
    to find the control field via commutative pop optimization to reach the taggert unitary.
    """
    def __init__(self, **kwargs):
        """
        Constructor
        :param kwargs: all the arguments are passed to the constructor of class CABSQCPolyOpt
        """
        super(CQCPolyOptMagnus, self).__init__(**kwargs)

        ################################################################################################################
        #
        #   Define the objective polynomial function using the Magnus series
        #
        ################################################################################################################

        # symbols
        t1, t2 = symbols('t1, t2', real=True)

        # introduce alias
        def h(t):
            return self.H0 + self.V * self.u(t)

        def Commutator(a, b):
            return a * b - b * a

        first_order = -integrate(h(t1), (t1, 0, self.T))

        second_order = -I * Rational(1, 2) * integrate(
            Commutator(integrate(h(t2), (t2, 0, t1)), h(t1)),
            (t1, 0, self.T)
        )

        magnus_series = first_order + second_order

        # notice no exponentiation (see the paper for details)
        self.obj = (
                (magnus_series - self.U_target).norm() ** 2
        ).evalf()

        ################################################################################################################
        #
        #   Find the control via optimization
        #
        ################################################################################################################

        self.get_control()