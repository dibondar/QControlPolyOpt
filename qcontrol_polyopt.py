"""
Library of classes for quantum control via commutative pop optimization
"""
from ncpol2sdpa import generate_variables, SdpRelaxation
from sympy import symbols, lambdify, re, exp, I, eye, integrate, Matrix, Rational

def commutator(a, b):
    return a @ b - b @ a

class QControlPolyOpt(object):
    """
    Quantum Control via Commutative Pop Opt with Unitary Preserving Pade Approximation for Magnus expansion
    """
    def __init__(self, *, H0, V, T=1, npoly=2, **kwargs):
        """
        Constructor
        :param H0: the Drift Hamiltonian
        :param V: Control Hamiltonian
        :param T: Terminal time
        :param kwargs: ignored
        :param npoly: (int) the degree of polynomial approximation for the sought control field
        """
        self.T = T

        # convert to sympy matrix
        self.H0 = Matrix(H0)
        self.V = Matrix(V)

        ################################################################################################################
        #
        # Define control $u(t)$
        #
        ################################################################################################################

        # Declare unknowns to be found
        self.x = generate_variables('x', npoly + 1)

        # Declare time variable
        self.t = symbols('t', real=True)

        Omega = self.get_truncated_Magnus_expansion(self.u)

        ident = self.ident = eye(Omega.shape[0])

        ################################################################################################################
        #
        # Evaluating Matrix exponent by unitarity preserving Pade approximation (i.e., the Cayley transform)
        # $$
        #     \exp \Omega = \frac{1 + \tanh(\Omega/2)}{1 - \tanh(\Omega/2)}, \\
        #     \tanh(\Omega/2) \approx \Omega/2 - \frac{1}{3}(\Omega/2)^3
        #     + \frac{2}{15} (\Omega/2)^5 \\
        #      = \Omega/2 \left(1 + (\Omega/2)^2 \left[-\frac{1}{3} +\frac{2}{15} (\Omega/2)^2 \right] \right)
        # $$
        #
        # In the next section we evaluate the above approximation for \tanh(\Omega/2) and save it as self.approx_tanh.
        # Note that we empirically found that it is good to keep first 3 terms in the Taylor expansion for tanh
        #
        ################################################################################################################

        Omega22 = (Omega @ Omega / 4).simplify()

        self.approx_tanh = Omega / 2 @ (
            ident + Omega22 @ (Rational(-1, 3) * ident + Rational(2, 15) * Omega22)
        )
        self.approx_tanh = self.approx_tanh.simplify()

    @property
    def u(self):
        """
        Postulated the polynomial shape for controls
        :return: lambdify simpy function
        """
        return lambdify(self.t, sum(c * self.t ** n_ for n_, c in enumerate(self.x)))

    def get_controls(self, U_target, spd_relax=5):
        """
        Find a control field synthesizing a specified unitary target
        :param U_target: (numpy.array) the target unitary gate to be synthesized
        :param relax_level: (int) SPD relaxation level
        :return: self. Results saved as properties self.u_opt, self.opt_x, self.obj_poly_val
        """
        #  Construct the polynomial objective function to be minimized
        self.obj_poly = (
            self.ident - U_target + (self.ident + U_target) @ self.approx_tanh
        ).norm() ** 2

        self.obj_poly = re(self.obj_poly)

        # perform minimization
        sdp = SdpRelaxation(self.x)
        sdp.get_relaxation(spd_relax, objective=self.obj_poly)
        sdp.solve(solver='mosek')

        # extract the values of control
        if sdp.status == 'optimal':
            self.opt_x = {_: sdp[_] for _ in sdp.variables}
            self.obj_poly_val = self.obj_poly.subs(self.opt_x)
            self.u_opt = lambdify(self.t, self.u(self.t).subs(self.opt_x))
        else:
            self.opt_x = self.u_opt = self.obj_poly_val = None

        return self

    def A(self, u, t):
        """
        The generator of motion entering the Magnus expansion
        :param u: control
        :param t:  time variable
        :return: sympy matrix
        """
        return (self.H0 + self.V * u(t)) / I

    def get_truncated_Magnus_expansion(self, u):
        """
        Return \Omega = \Omega_1 + \Omega_2 - the partial sum of the Magnus expansion
        :param u: control as lambdify function of time
        :return: sympy matrix
        """
        # Declare time variables
        t1, t2, t3 = symbols('t1, t2, t3', real=True)

        A1 = self.A(u, t1)
        A2 = self.A(u, t2)
        A3 = self.A(u, t3)

        T = self.T

        Omega1 = integrate(A1, (t1, 0, T))

        Omega2 = Rational(1, 2) * integrate(integrate(
            commutator(A1, A2),
            (t2, 0, t1)), (t1, 0, T)
        )

        # We empirically found that adding 3-rd order term does not improve much results

        # Omega3 = Rational(1, 6) * integrate(integrate(integrate(
        #    commutator(A1, commutator(A2, A3))
        #    + commutator(commutator(A1, A2), A3),
        #    (t3, 0, t2)), (t2, 0, t1)), (t1, 0, T)
        # )

        return Omega1 + Omega2 # + Omega3
