""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

""" This module contains the approximate safety filters for the nonlinear system"""
from typing import Dict

import casadi
import torch
from pytope import Polytope

from apcbf.approximator import *
from apcbf.controller import *
from apcbf.dynamic import *
from apcbf.nn_model_types import NonLinModelType
from apcbf.non_lin_sys import *


def output_modifier(h):
    return np.exp(h) - 1


class APCBFSafetyFilterKINFDEC(Controller):
    def __init__(self, model, m_type: NonLinModelType, use_log_space_model: bool,
                 use_plus_model: bool, sys: NonLinearDiscreteDynamics, params_dict: Dict,
                 input_constraints: Polytope, performance_controller: Controller, zero_tol: float = 1e-5, kinf_factor: float = 0.5, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self.model = model
        self.zero_tol = zero_tol
        self.performance_ctrl = performance_controller
        self.T_disc = params_dict['T_disc']
        T_disc = self.T_disc
        self.input_dim = sys.input_dim
        self.state_dim = sys.state_dim
        self.U = input_constraints
        self.lower_u = -self.U.b[1::2]
        self.upper_u = self.U.b[::2]
        self.use_log_space_model = use_log_space_model
        self.kinf_factor = kinf_factor

        L = 5.0
        vs = 5.0

        self.weights = []
        self.bias = []
        for layer in model.children():
            self.weights.append(layer.state_dict()['weight'])
            self.bias.append(layer.state_dict()['bias'])

        # Define Parameter
        self.p = casadi.MX.sym("p", (self.state_dim, 1))
        self.h_current = casadi.MX.sym("hc")
        self.delta_h = casadi.MX.sym("dh")
        self.u_p = casadi.MX.sym("up", (self.input_dim, 1))

        # Define Optimization variable
        self.u = casadi.MX.sym("u", (self.input_dim, 1))
        self.xa1 = self.p[0] + self.T_disc * \
            (vs + self.p[3])*casadi.sin(self.p[1])
        self.xa2 = self.p[1] + self.T_disc * \
            ((vs + self.p[3])/L)*casadi.tan(self.p[2])
        self.xa3 = self.p[2] + self.T_disc * self.u[0]
        self.xa4 = self.p[3] + self.T_disc * self.u[1]
        self.x = casadi.vertcat(self.xa1, self.xa2, self.xa3, self.xa4)

        self.xi = casadi.MX.sym("xi")

        # Casadi <-> NN model
        # Two hidden layers architecture
        if m_type == NonLinModelType.TWOHIDDENLAYERS or m_type == NonLinModelType.A or m_type == NonLinModelType.APLUS:
            self.x1 = self.weights[0].numpy()@self.x + \
                self.bias[0].numpy()  # First layer
            # self.x1out = fmax(self.x1,0) # relu
            self.x1out = casadi.log(1 + casadi.exp(self.x1))  # softplus
            # self.x1out = 1 / (1 + casadi.exp(-self.x1)) #sigmoid

            self.x2 = self.weights[1].numpy()@self.x1out + \
                self.bias[1].numpy()  # Second Layer
            # self.x2out = fmax(self.x2,0) # relu
            self.x2out = casadi.log(1 + casadi.exp(self.x2))  # softplus
            # self.x2out = 1 / (1 + casadi.exp(-self.x2)) #sigmoid

            self.fout1 = self.weights[2].numpy(
            )@self.x2out + self.bias[2].numpy()  # Third Layer

        # Three hidden layers architecture
        if m_type == NonLinModelType.THREEHIDDENLAYERS or m_type == NonLinModelType.B or m_type == NonLinModelType.C or m_type == NonLinModelType.BPLUS or m_type == NonLinModelType.CPLUS or m_type == NonLinModelType.GPLUS:
            # NN Version B also C
            self.x1 = self.weights[0].numpy()@self.x + \
                self.bias[0].numpy()  # First layer
            self.x1out = casadi.log(1 + casadi.exp(self.x1))  # softplus

            self.x2 = self.weights[1].numpy()@self.x1out + \
                self.bias[1].numpy()  # Second Layer
            self.x2out = casadi.log(1 + casadi.exp(self.x2))  # softplus

            self.x3 = self.weights[2].numpy()@self.x2out + \
                self.bias[2].numpy()  # Second Layer
            self.x3out = casadi.log(1 + casadi.exp(self.x3))  # softplus

            self.fout1 = self.weights[3].numpy(
            )@self.x3out + self.bias[3].numpy()  # Third Layer

        # Four hidden layers
        if m_type == NonLinModelType.FOURHIDDENLAYERS or m_type == NonLinModelType.D or m_type == NonLinModelType.DPLUS:
            self.x1 = self.weights[0].numpy()@self.x + \
                self.bias[0].numpy()  # First layer
            self.x1out = casadi.log(1 + casadi.exp(self.x1))  # softplus

            self.x2 = self.weights[1].numpy()@self.x1out + \
                self.bias[1].numpy()  # Second Layer
            self.x2out = casadi.log(1 + casadi.exp(self.x2))  # softplus

            self.x3 = self.weights[2].numpy()@self.x2out + \
                self.bias[2].numpy()  # Second Layer
            self.x3out = casadi.log(1 + casadi.exp(self.x3))  # softplus

            self.x4 = self.weights[3].numpy()@self.x3out + \
                self.bias[3].numpy()  # Third Layer
            self.x4out = casadi.log(1 + casadi.exp(self.x4))  # softplus

            self.fout1 = self.weights[4].numpy(
            )@self.x4out + self.bias[4].numpy()  # Third Layer

        # Plus version have soft plus output
        if use_plus_model:
            self.fout = casadi.log(1 + casadi.exp(self.fout1))  # softplus
        else:
            self.fout = self.fout1

        # If use log space model need to calc. back to original space
        if use_log_space_model:
            self.fout = casadi.exp(self.fout) - 1

        if not self.verbose:
            # disable output, TODO need way of obtaining status
            self.opts = {'ipopt.print_level': 0, 'print_time': 0}
        else:
            self.opts = {}

        # Objective and constraint formulation
        self.alpha = 1e4  # Slack penalty
        self.f2 = (self.u_p - self.u).T @ (self.u_p -
                                           self.u) + self.alpha * self.xi

        self.g2_1 = self.fout - self.h_current + self.delta_h - \
            self.zero_tol - self.xi
        self.g2_2 = -self.xi

        self.solver2 = casadi.nlpsol("solver", "ipopt", {'x': casadi.vertcat(self.u, self.xi), 'p': casadi.vertcat(self.p, self.h_current, self.u_p, self.delta_h),
                                                         'f': self.f2, 'g': casadi.vertcat(self.g2_1, self.g2_2)}, self.opts)
        self.xi_list = []
        self.hpb_traj = []

    def __str__(self):
        return 'APCBF_SF_KINFDEC'

    def input(self, x):
        # Obtain performance controller input
        u_p = self.performance_ctrl.input(x)
        # Evaluate approximate CBF at x(k)
        with torch.no_grad():
            h_x = self.model(torch.tensor(
                x).reshape(-1, self.state_dim).float()).numpy()
            if self.use_log_space_model:
                h_x = output_modifier(h_x)

        self.hpb_traj.append(h_x[0])

        delta_h = self.kinf_factor * h_x

        res2 = self.solver2(x0=casadi.vertcat([0.0, 0.0], 0.0), p=casadi.vertcat(
            x, h_x, u_p, delta_h), lbx=casadi.vertcat(self.lower_u, 0), ubx=casadi.vertcat(self.upper_u, 1e6), ubg=0)
        u = res2['x'].full()[:self.input_dim]
        xi = res2['x'].full()[self.input_dim]

        self.xi_list.append(xi)

        return u.flatten()


class APCBFSafetyFilterMAXDEC(Controller):
    def __init__(self, model, m_type: NonLinModelType, use_log_space_model, use_plus_model, sys, params_dict, input_constraints, performance_controller, zero_tol=1e-5, verbose=False) -> None:
        super().__init__()
        self.verbose = verbose
        self.model = model
        self.zero_tol = zero_tol
        self.performance_ctrl = performance_controller
        self.T_disc = params_dict['T_disc']
        T_disc = self.T_disc
        self.input_dim = sys.input_dim
        self.state_dim = sys.state_dim
        self.U = input_constraints
        self.lower_u = -self.U.b[1::2]
        self.upper_u = self.U.b[::2]
        self.use_log_space_model = use_log_space_model

        L = 5.0
        vs = 5.0
        # Extract weights for neural net model
        self.weights = []
        self.bias = []
        for layer in self.model.children():
            self.weights.append(layer.state_dict()['weight'])
            self.bias.append(layer.state_dict()['bias'])

        # Define Parameter
        self.p = casadi.MX.sym("p", (self.state_dim, 1))
        self.h_current = casadi.MX.sym("hc")
        self.delta_h = casadi.MX.sym("dh")
        self.u_p = casadi.MX.sym("up", (self.input_dim, 1))

        # Optimization variable
        self.u = casadi.MX.sym("u", (self.input_dim, 1))
        self.xa1 = self.p[0] + self.T_disc * \
            (vs + self.p[3])*casadi.sin(self.p[1])
        self.xa2 = self.p[1] + self.T_disc * \
            ((vs + self.p[3])/L)*casadi.tan(self.p[2])
        self.xa3 = self.p[2] + self.T_disc * self.u[0]
        self.xa4 = self.p[3] + self.T_disc * self.u[1]
        self.x = casadi.vertcat(self.xa1, self.xa2, self.xa3, self.xa4)
        # self.xi = casadi.MX.sym("xi") # Slack variable

        # Casadi <-> NN model
        # Two hidden layers neural network
        if m_type == NonLinModelType.TWOHIDDENLAYERS or m_type == NonLinModelType.A or m_type == NonLinModelType.APLUS:
            self.x1 = self.weights[0].numpy()@self.x + \
                self.bias[0].numpy()  # First layer
            # self.x1out = fmax(self.x1,0) # relu
            self.x1out = casadi.log(1 + casadi.exp(self.x1))  # softplus
            # self.x1out = 1 / (1 + casadi.exp(-self.x1)) #sigmoid

            self.x2 = self.weights[1].numpy()@self.x1out + \
                self.bias[1].numpy()  # Second Layer
            # self.x2out = fmax(self.x2,0) # relu
            self.x2out = casadi.log(1 + casadi.exp(self.x2))  # softplus
            # self.x2out = 1 / (1 + casadi.exp(-self.x2)) #sigmoid

            self.fout1 = self.weights[2].numpy(
            )@self.x2out + self.bias[2].numpy()  # Third Layer

        # Three hidden layers neural network
        if m_type == NonLinModelType.THREEHIDDENLAYERS or m_type == NonLinModelType.B or m_type == NonLinModelType.C or m_type == NonLinModelType.BPLUS or m_type == NonLinModelType.CPLUS or m_type == NonLinModelType.GPLUS:
            # NN Version B also C
            self.x1 = self.weights[0].numpy()@self.x + \
                self.bias[0].numpy()  # First layer
            self.x1out = casadi.log(1 + casadi.exp(self.x1))  # softplus

            self.x2 = self.weights[1].numpy()@self.x1out + \
                self.bias[1].numpy()  # Second Layer
            self.x2out = casadi.log(1 + casadi.exp(self.x2))  # softplus

            self.x3 = self.weights[2].numpy()@self.x2out + \
                self.bias[2].numpy()  # Second Layer
            self.x3out = casadi.log(1 + casadi.exp(self.x3))  # softplus

            self.fout1 = self.weights[3].numpy(
            )@self.x3out + self.bias[3].numpy()  # Third Layer

        # Four hidden layers neural network
        if m_type == NonLinModelType.FOURHIDDENLAYERS or m_type == NonLinModelType.D or m_type == NonLinModelType.DPLUS:
            self.x1 = self.weights[0].numpy()@self.x + \
                self.bias[0].numpy()  # First layer
            self.x1out = casadi.log(1 + casadi.exp(self.x1))  # softplus

            self.x2 = self.weights[1].numpy()@self.x1out + \
                self.bias[1].numpy()  # Second Layer
            self.x2out = casadi.log(1 + casadi.exp(self.x2))  # softplus

            self.x3 = self.weights[2].numpy()@self.x2out + \
                self.bias[2].numpy()  # Second Layer
            self.x3out = casadi.log(1 + casadi.exp(self.x3))  # softplus

            self.x4 = self.weights[3].numpy()@self.x3out + \
                self.bias[3].numpy()  # Third Layer
            self.x4out = casadi.log(1 + casadi.exp(self.x4))  # softplus

            self.fout1 = self.weights[4].numpy(
            )@self.x4out + self.bias[4].numpy()  # Third Layer

        # Plus version have soft plus output
        if use_plus_model:
            self.fout = casadi.log(1 + casadi.exp(self.fout1))  # softplus
        else:
            self.fout = self.fout1

        # If use log space model need to calc. back to original space
        if use_log_space_model:
            self.fout = casadi.exp(self.fout) - 1
        if not self.verbose:
            self.opts = {'ipopt.print_level': 0, 'print_time': 0}
        else:
            self.opts = {}

        # Objective and constraint formulation
        self.f1 = self.fout
        self.f2 = (self.u_p - self.u).T @ (self.u_p -
                                           self.u)  # +  self.alpha * self.xi

        # self.g2_1 = self.fout - self.h_current - self.delta_h - self.zero_tol
        self.g2_1 = self.fout - self.delta_h
        # self.g2_2 = -self.xi

        self.solver1 = casadi.nlpsol(
            "solver", "ipopt", {'x': self.u, 'p': self.p, 'f': self.f1}, self.opts)
        self.solver2 = casadi.nlpsol("solver", "ipopt", {'x': self.u, 'p': casadi.vertcat(
            self.p, self.h_current, self.delta_h, self.u_p), 'f': self.f2, 'g': self.g2_1}, self.opts)

        self.xi_list = []
        self.hpb_traj = []
        self.min_decrease_list = []

    def __str__(self):
        return 'APCBF_SF_MAXDEC'

    def input(self, x):
        # Obtain performance controller input
        u_p = self.performance_ctrl.input(x)
        # Evaluate approximate CBF at x(k)
        with torch.no_grad():
            h_x = self.model(torch.tensor(
                x).reshape(-1, self.state_dim).float()).numpy()
            if self.use_log_space_model:
                h_x = output_modifier(h_x)

        self.hpb_traj.append(h_x[0])

        # 1. opt. problem - max decrease computation
        x = x.reshape((-1, 1))
        res1 = self.solver1(x0=0.0, p=x, lbx=self.lower_u, ubx=self.upper_u)
        delta_h = res1['f'].full()[0]  # - h_x[0]
        max_dec_u = res1['x'].full()[0]
        self.min_decrease_list.append(delta_h)

        # 2. opt. problem - safety filter using max decrease
        # init_u = [0.0,0.0]
        init_u = max_dec_u
        res2 = self.solver2(x0=init_u, p=casadi.vertcat(
            x, h_x, delta_h, u_p), lbx=self.lower_u, ubx=self.upper_u, ubg=[0.0])
        u = res2['x'].full()[:self.input_dim].flatten()
        # xi = res2['x'].full()[self.input_dim]

        # self.xi_list.append(xi)
        # if self.verbose :
        # print("succes ? ", self.solver.stats()['success'])

        return u
