""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

""" This module implements the PCBF-SF for the nonlinear system"""
import casadi
import numpy as np

from apcbf.controller import *
from apcbf.dynamic import NonLinearDiscreteDynamics
from typing import Dict, Callable
from pytope import Polytope


class SlackOpt():
    """ This class implements the optimal slack computation step"""

    def __init__(self, sys: NonLinearDiscreteDynamics, X: Polytope, U: Polytope, delta_i: Callable, param_dict: Dict, N: int = 50, use_warm_start: bool = False, verbose: bool = False):
        self.N = N
        self.verbose = verbose
        self.constraint_tol = 0.0
        self.use_warm_start = use_warm_start

        vs = 5.0
        L = 5.0

        self.n = sys.state_dim
        self.m = sys.input_dim
        self.numb_x_constr = X.b.shape[0]

        self.P_f = param_dict['P_f']
        self.gamma_x = param_dict['gamma_x']
        self.gamma_f = param_dict['gamma_f']
        self.alpha_f = param_dict['alpha_f']
        self.T_disc = param_dict['T_disc']

        self.opti = casadi.Opti()

        self.x0 = self.opti.parameter(self.n, 1)
        self.x = self.opti.variable(self.n, self.N + 1)
        self.u = self.opti.variable(self.m, self.N)
        self.gsi = self.opti.variable(X.b.shape[0], self.N)
        self.gsi_N = self.opti.variable()

        self.obj = self.alpha_f * self.gsi_N
        for i in range(self.N):
            self.obj += casadi.norm_2(self.gsi[:, i])

        self.opti.minimize(self.obj)

        # Constraints
        self.constraints = [self.x[:, 0] == self.x0]

        for t in range(self.N):
            self.constraints.append(
                self.x[0, t + 1] == self.x[0, t] + self.T_disc * (vs + self.x[3, t]) * casadi.sin(self.x[1, t]))
            self.constraints.append(
                self.x[1, t + 1] == self.x[1, t] + self.T_disc * ((vs + self.x[3, t]) / L) * casadi.tan(self.x[2, t]))
            self.constraints.append(
                self.x[2, t + 1] == self.x[2, t] + self.T_disc * self.u[0, t])
            self.constraints.append(
                self.x[3, t + 1] == self.x[3, t] + self.T_disc * self.u[1, t])

            self.constraints.append(X.A @ self.x[:, t] <= X.b - delta_i(t) * np.ones(X.b.shape) + self.gsi[:,
                                                                                                           t] - self.constraint_tol)  # tightening + relaxation + tolerance
            self.constraints.append(
                U.A @ self.u[:, t] <= U.b - self.constraint_tol)
            self.constraints.append(self.gsi[:, t] >= 0)

        # Terminal constraints
        self.constraints.append(self.gsi_N >= 0)
        self.constraints.append(self.x[:, self.N].T @ self.P_f @ self.x[:,
                                                                        self.N] - self.gamma_x <= self.gsi_N - self.constraint_tol)

        self.opti.subject_to(self.constraints)

        if not self.verbose:
            self.opts = {'ipopt.print_level': 0,
                         'print_time': 0}  # disable output
            self.opti.solver('ipopt', self.opts)
        else:
            self.opti.solver('ipopt')

    def solve(self, x):
        """ Returns the computed optimal slack variables together with the optimal value function.
        In additon, it also returns the optimal input and state trajectory which can be used to warm start
        the next optimziaton problem"""
        self.opti.set_value(self.x0, x.reshape((self.n, 1)))

        # Init gsi_temp with nonzero value othw. get ipopt error (Gradients nan etc.)
        if not self.use_warm_start:
            gsi_temp = 1 * np.ones((self.numb_x_constr, self.N))
            self.opti.set_initial(self.gsi, gsi_temp)

        self.sol = self.opti.solve()
        return self.sol.value(self.gsi), self.sol.value(self.gsi_N), self.sol.value(self.obj), self.sol.value(
            self.u), self.sol.value(self.x)


class SafetyFilter():
    """ This class implements the safety filter computation step of the PCBF-SF scheme"""

    def __init__(self, sys: NonLinearDiscreteDynamics, performance_ctrl: Controller,
                 X: Polytope, U: Polytope, delta_i: Callable, param_dict: Dict, N: int = 50, verbose: bool = False):
        self.N = N
        self.verbose = verbose
        self.constraint_tol = 0.0
        self.perf_ctrl = performance_ctrl

        vs = 5.0
        L = 5.0

        self.n = sys.state_dim
        self.m = sys.input_dim
        u_shape = (self.m, 1)
        x_shape = (self.n, 1)

        self.P_f = param_dict['P_f']
        self.gamma_x = param_dict['gamma_x']
        self.gamma_f = param_dict['gamma_f']
        self.alpha_f = param_dict['alpha_f']
        self.T_disc = param_dict['T_disc']

        self.opti = casadi.Opti()

        self.x0 = self.opti.parameter(self.n, 1)
        self.ul = self.opti.parameter(self.m, 1)

        self.gsi = self.opti.parameter(X.b.shape[0], self.N)
        self.gsi_N = self.opti.parameter()

        self.x = self.opti.variable(self.n, self.N + 1)
        self.u = self.opti.variable(self.m, self.N)

        # Objective (Safety Filter)
        self.obj = (self.ul - self.u[:, 0]).T @ (self.ul - self.u[:, 0])

        # Constraints
        self.constraints = [self.x[:, 0] == self.x0]

        for t in range(self.N):
            self.constraints.append(
                self.x[0, t + 1] == self.x[0, t] + self.T_disc * (vs + self.x[3, t]) * casadi.sin(self.x[1, t]))
            self.constraints.append(
                self.x[1, t + 1] == self.x[1, t] + self.T_disc * ((vs + self.x[3, t]) / L) * casadi.tan(self.x[2, t]))
            self.constraints.append(
                self.x[2, t + 1] == self.x[2, t] + self.T_disc * self.u[0, t])
            self.constraints.append(
                self.x[3, t + 1] == self.x[3, t] + self.T_disc * self.u[1, t])

            self.constraints.append(X.A @ self.x[:, t] <= X.b - delta_i(t) * np.ones(X.b.shape) + self.gsi[:,
                                                                                                           t] - self.constraint_tol)  # tightening + relaxation + tolerance
            self.constraints.append(
                U.A @ self.u[:, t] <= U.b - self.constraint_tol)

        # Terminal constraints
        self.constraints.append(self.x[:, self.N].T @ self.P_f @ self.x[:,
                                                                        self.N] - self.gamma_x <= self.gsi_N - self.constraint_tol)

        self.opti.subject_to(self.constraints)
        self.opti.minimize(self.obj)

        if not self.verbose:
            self.opts = {'ipopt.print_level': 0,
                         'print_time': 0}  # disable output
            self.opti.solver('ipopt', self.opts)
        else:
            self.opti.solver('ipopt')

    def input(self, x, gsi_N, gsi, u_init=None, x_init=None):
        """ Returns a guaranteed safe input which is able to recover states outside the state constraints (but still
        inside the domain of h_PB"""
        ul = self.perf_ctrl.input(x)
        success = False

        # Warm start if available
        if u_init is not None:
            self.opti.set_initial(self.u, u_init)
        if x_init is not None:
            self.opti.set_initial(self.x, x_init)

        self.opti.set_value(self.x0, x.reshape((self.n, 1)))
        self.opti.set_value(self.ul, ul.reshape((self.m, 1)))
        self.opti.set_value(self.gsi, gsi)
        self.opti.set_value(self.gsi_N, gsi_N)

        self.sol = self.opti.solve()
        success = True
        return self.sol.value(self.u)[:, 0], success


class Algorithm(Controller):
    """ This class combines the slack variables computation step with the safety filter optimziation scheme to get Algorithm 1 """

    def __init__(self, sys: NonLinearDiscreteDynamics, performance_ctrl: Controller,
                 X: Polytope, U: Polytope, delta_i: Callable, param_dict: Dict, N: int=50, verbose: bool=False):
        self.slack_opt = SlackOpt(
            sys, X, U, delta_i, param_dict, N=N, verbose=verbose)
        self.safety_filter = SafetyFilter(
            sys, performance_ctrl, X, U, delta_i, param_dict, N=N, verbose=verbose)
        self.performance_controller = performance_ctrl
        self.verbose = verbose
        self.hpb_traj = []  # For debugging purposes store h_PB trajectory

    def input(self, x):
        if self.verbose:
            print(f"@x={x}")
        # Step 1 : Compute optimal slack variables
        gsi, gsi_N, hbp, u_init, x_init = self.slack_opt.solve(x)
        self.ustar = u_init
        self.hpb_traj.append(hbp)
        # Step 2 : Compute safety filter problem
        u, success = self.safety_filter.input(x, gsi_N, gsi, u_init, x_init)

        return u

    def reset(self):
        self.hpb_traj = []
