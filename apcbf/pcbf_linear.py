""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

""" This module implements the PCBF-SF scheme from Wabersich et. al 2022 for a linear system"""
from typing import Callable

import cvxpy as cp
import numpy as np
from pytope import Polytope

from apcbf.controller import *
from apcbf.dynamic import LinearDiscreteDynamics

#OPTIMIZER = cp.MOSEK
OPTIMIZER = cp.ECOS


class SlackOpt():
    """ Slack Computation step for the safety filter with recovery mechanism for a generic linear system"""

    def __init__(self, lin_sys: LinearDiscreteDynamics, X: Polytope, U: Polytope, delta_i: Callable,
                 alpha_f: float, P_f: np.matrix, gamma_x: float, N: int = 20, verbose: bool = False, tol: float = 1e-6):
        self.N = N
        self.verbose = verbose
        self.constraint_tol = tol

        A = lin_sys.A
        self.n = lin_sys.state_dim
        B = lin_sys.B
        self.m = lin_sys.input_dim

        self.x0 = cp.Parameter((self.n, 1))
        self.x = cp.Variable((self.n, self.N + 1))
        self.u = cp.Variable((self.m, self.N))
        self.gsi = cp.Variable((X.b.shape[0], self.N))
        self.gsi_N = cp.Variable()

        u_shape = (self.m, 1)
        x_shape = (self.n, 1)

        # Objective
        self.obj = cp.Minimize(alpha_f * self.gsi_N +
                               sum(cp.norm(self.gsi, axis=0)))

        # Constraints
        self.constraints = [cp.reshape(self.x[:, 0], x_shape) == self.x0]

        for t in range(self.N):
            self.constraints.append(
                self.x[:, t + 1] == A @ self.x[:, t] + B @ self.u[:, t])
            self.constraints.append(
                X.A @ cp.reshape(self.x[:, t], x_shape) <= X.b - delta_i(t) * np.ones(X.b.shape) + cp.reshape(
                    self.gsi[:, t], X.b.shape) - self.constraint_tol)  # tightening + relaxation + tolerance
            self.constraints.append(
                U.A @ cp.reshape(self.u[:, t], u_shape) <= U.b - self.constraint_tol)
            self.constraints.append(self.gsi[:, t] >= 0)

        # Terminal constraints
        self.constraints.append(self.gsi_N >= 0)
        self.constraints.append(cp.quad_form(self.x[:, self.N],
                                             P_f) - gamma_x <= self.gsi_N - self.constraint_tol)

        # Problem definition
        self.problem = cp.Problem(self.obj, self.constraints)

    def solve(self, x):
        """ Returns the computed optimal slack variables together with the optimal value function"""
        self.x0.value = x.reshape((self.n, 1))

        self.problem.solve(OPTIMIZER, verbose=self.verbose)

        if self.problem.status not in ["infeasible", "unbounded"]:
            if self.verbose > 0:
                print("Found optimal slack variables")
            return self.gsi.value, self.gsi_N.value, self.problem.value
        else:
            print("infeasible :", self.problem.status)
            raise Exception()


class SafetyFilter():
    """ This class implements the safety filter computation step of the PCBF-SF scheme"""

    def __init__(self, lin_sys: LinearDiscreteDynamics, performance_ctrl: Controller,
                 X: Polytope, U: Polytope, P_f: np.matrix, gamma_x: float, delta_i: Callable, N: int = 20, verbose: bool = False):
        self.verbose = verbose
        self.N = N
        A = lin_sys.A
        self.n = lin_sys.state_dim
        B = lin_sys.B
        self.m = lin_sys.input_dim

        self.perf_ctrl = performance_ctrl

        # Define optimization variables
        self.x = cp.Variable((self.n, self.N + 1))
        self.u = cp.Variable((self.m, self.N))

        self.x0 = cp.Parameter((self.n, 1))
        self.ul = cp.Parameter((self.m, 1))

        self.gsi = cp.Parameter((X.b.shape[0], self.N))
        self.gsi_N = cp.Parameter()

        self.min_expr = cp.quad_form(
            (self.ul - cp.reshape(self.u[:, 0], (self.m, 1))), np.eye(self.m))
        self.obj = cp.Minimize(self.min_expr)

        x_shape = self.x0.shape
        u_shape = self.ul.shape

        # Define constraints
        self.constraints = [cp.reshape(self.x[:, 0], x_shape) == self.x0]
        for t in range(self.N):
            self.constraints.append(
                self.x[:, t + 1] == A @ self.x[:, t] + B @ self.u[:, t])
            self.constraints.append(
                X.A @ cp.reshape(self.x[:, t], x_shape) <= X.b - delta_i(t) * np.ones(X.b.shape) + cp.reshape(
                    self.gsi[:, t], X.b.shape))  # tightening + relaxation
            self.constraints.append(
                U.A @ cp.reshape(self.u[:, t], u_shape) <= U.b)

        # Terminal constraints
        self.constraints.append(cp.quad_form(
            self.x[:, self.N], P_f) - gamma_x <= self.gsi_N)
        self.problem = cp.Problem(self.obj, self.constraints)

    def input(self, x, gsi_N, gsi):
        """ Returns a guaranteed safe input which is able to recover states outside the state constraints (but still
        inside the domain of h_PB"""

        # Compute input coming from performance controller
        ul = self.perf_ctrl.input(x)
        success = False

        if OPTIMIZER is cp.MOSEK:
            # For numerical reason MOSEK may return slack variables which are slightly negative, clip them to 0
            if gsi_N < 0:
                gsi_N = 0

        self.ul.value = ul.reshape((self.m, 1))
        self.x0.value = x.reshape((self.n, 1))
        self.gsi.value = gsi
        self.gsi_N.value = gsi_N

        self.problem.solve(OPTIMIZER,
                           verbose=self.verbose)  # ,mosek_params={mosek.iparam.infeas_report_auto : mosek.onoffkey.on})#, save_file='second_opt.opf')

        if self.problem.status not in ["infeasible", "unbounded"]:
            if self.verbose:
                print(
                    f" Found : Safe input : {self.u.value[:, 0]}, Performance input : {self.ul.value}")
            u = self.u.value[:, 0]
            success = True
        else:
            print(self.problem.status)
            raise Exception()

        return u, success


class Algorithm(Controller):
    """ This class combines the slack variables computation step with the safety filter optimziation scheme to get Algorithm 1 of the paper """

    def __init__(self, lin_sys: LinearDiscreteDynamics, performance_ctrl: Controller,
                 X: Polytope, U: Polytope, P_f: np.matrix, alpha_f: float, gamma_x: float, delta_i: Callable, N: int = 20, verbose: bool = False, ):

        self.slack_opt = SlackOpt(
            lin_sys, X, U, delta_i, alpha_f, P_f, gamma_x, verbose=verbose, N=N)
        self.safety_filter = SafetyFilter(
            lin_sys, performance_ctrl, X, U, P_f, gamma_x, delta_i, verbose=verbose, N=N)
        self.performance_controller = performance_ctrl
        self.verbose = verbose
        self.hpb_traj = []  # For visualization purposes store h_PB trajectory

    def input(self, x):
        if self.verbose:
            print(f"@x={x}")
        # Compute optimal slack variables
        gsi, gsi_N, hbp = self.slack_opt.solve(x)
        self.hpb_traj.append(hbp)
        # Compute safety filter optimizaiton problem using computed slack variables
        u, success = self.safety_filter.input(x, gsi_N, gsi)
        return u

    def reset(self):
        self.hpb_traj = []
