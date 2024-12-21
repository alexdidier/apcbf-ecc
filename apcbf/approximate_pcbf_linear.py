""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

""" This module contains the approximate safety filters for the linear case"""


from typing import Dict

import casadi
import torch

from apcbf.controller import *
from apcbf.dynamic import *

from .nn_model_types import *


class APCBFSafetyFilterKINFDEC(Controller):
    def __init__(self, model: torch.nn.Module, m_type: LinModelType, lin_sys: LinearDiscreteDynamics,
                 performance_controller: Controller, params_dict: Dict,  zero_tol: float = 1e-4, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self.model = model
        self.zero_tol = zero_tol
        self.performance_ctrl = performance_controller
        self.A = lin_sys.A
        self.B = lin_sys.B
        self.state_dim = lin_sys.state_dim
        self.U = params_dict["U"]
        self.lower_u = -self.U.b[1::2]
        self.upper_u = self.U.b[::2]

        self.weights = []
        self.bias = []

        self.input_dim = lin_sys.input_dim
        self.state_dim = lin_sys.state_dim

        for layer in self.model.children():
            self.weights.append(layer.state_dict()['weight'])
            self.bias.append(layer.state_dict()['bias'])

        # Parameter
        self.p = casadi.MX.sym("p", (self.state_dim, 1))
        self.h_current = casadi.MX.sym("hc")
        self.delta_h = casadi.MX.sym("dh")
        self.u_p = casadi.MX.sym("u", (self.input_dim, 1))

        # Optimization variable
        self.u = casadi.MX.sym("u", (self.input_dim, 1))
        self.x = casadi.MX(lin_sys.A @ self.p) + lin_sys.B @ self.u
        self.xi = casadi.MX.sym("xi")

        # NN Version A
        if m_type == LinModelType.A or m_type == LinModelType.APLUS:
            self.x1 = self.weights[0].numpy() @ self.x + \
                self.bias[0].numpy()  # First layer
            # self.x1out = fmax(self.x1,0) # relu
            self.x1out = casadi.log(1 + casadi.exp(self.x1))  # softplus
            # self.x1out = 1 / (1 + casadi.exp(-self.x1)) #sigmoid

            self.x2 = self.weights[1].numpy(
            ) @ self.x1out + self.bias[1].numpy()  # Second Layer
            # self.x2out = fmax(self.x2,0) # relu
            self.x2out = casadi.log(1 + casadi.exp(self.x2))  # softplus
            # self.x2out = 1 / (1 + casadi.exp(-self.x2)) #sigmoid

            self.fout1 = self.weights[2].numpy(
            ) @ self.x2out + self.bias[2].numpy()  # Third Layer

        if m_type == LinModelType.B or m_type == LinModelType.C or m_type == LinModelType.BPLUS or m_type == LinModelType.CPLUS:
            # NN Version B also C
            self.x1 = self.weights[0].numpy() @ self.x + \
                self.bias[0].numpy()  # First layer
            self.x1out = casadi.log(1 + casadi.exp(self.x1))  # softplus

            self.x2 = self.weights[1].numpy(
            ) @ self.x1out + self.bias[1].numpy()  # Second Layer
            self.x2out = casadi.log(1 + casadi.exp(self.x2))  # softplus

            self.x3 = self.weights[2].numpy(
            ) @ self.x2out + self.bias[2].numpy()  # Second Layer
            self.x3out = casadi.log(1 + casadi.exp(self.x3))  # softplus

            self.fout1 = self.weights[3].numpy(
            ) @ self.x3out + self.bias[3].numpy()  # Third Layer

        # Plus version have soft plus output
        if m_type == LinModelType.APLUS or m_type == LinModelType.BPLUS or m_type == LinModelType.CPLUS:
            self.fout = casadi.log(1 + casadi.exp(self.fout1))  # softplus
        else:
            self.fout = self.fout1

        if not self.verbose:
            self.opts = {'ipopt.print_level': 0,
                         'print_time': 0}  # disable output,
        else:
            self.opts = {}

        # Objective and constraint formulation
        self.alpha = 1e5  # Slack penalty
        self.f2 = (self.u_p - self.u).T @ (self.u_p -
                                           self.u) + self.alpha * self.xi

        self.g2_1 = self.fout - self.h_current + self.delta_h - \
            self.zero_tol - self.xi
        self.g2_2 = -self.xi

        self.solver = casadi.nlpsol("solver", "ipopt", {'x': casadi.vertcat(self.u, self.xi), 'p': casadi.vertcat(self.p, self.h_current, self.u_p, self.delta_h),
                                                        'f': self.f2, 'g': casadi.vertcat(self.g2_1, self.g2_2)}, self.opts)
        self.xi_list = []
        self.hpb_traj = []

    def input(self, x):
        # Obtain performance controller input
        u_p = self.performance_ctrl.input(x)
        # Evaluate approximate CBF at x(k)
        with torch.no_grad():
            h_x = self.model(torch.tensor(
                x).reshape(-1, self.state_dim).float()).numpy()

            # Don't use log space models for linear dynamics
            # if self.use_log_space_model:
            #     h_x = output_modifier(h_x)

        self.hpb_traj.append(h_x[0])
        delta_h = 0.5 * h_x

        res2 = self.solver(x0=casadi.vertcat([0.0, 0.0]), p=casadi.vertcat(
            x, h_x, u_p, delta_h), lbx=casadi.vertcat(self.lower_u, 0), ubx=casadi.vertcat(self.upper_u, 1e6), ubg=0)
        u = res2['x'].full()[:self.input_dim]
        xi = res2['x'].full()[self.input_dim]

        self.xi_list.append(xi)

        #
        # if self.verbose :
        #    print("succes ? ", self.solver.stats()['success'])

        return u.flatten()


class APCBFSafetyFilterMAXDEC(Controller):
    """ This class implements Approach 2 for a generic linear system using the trained neural network model"""

    def __init__(self, model: torch.nn.Module, m_type: LinModelType, lin_sys: LinearDiscreteDynamics,
                 performance_controller: Controller, params_dict: Dict, zero_tol: float = 1e-6, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self.model = model
        self.zero_tol = zero_tol
        self.performance_ctrl = performance_controller

        self.state_dim = lin_sys.state_dim
        self.weights = []
        self.bias = []
        self.U = params_dict["U"]
        self.lower_u = -self.U.b[1::2]
        self.upper_u = self.U.b[::2]

        self.input_dim = lin_sys.input_dim
        self.state_dim = lin_sys.state_dim

        for layer in self.model.children():
            self.weights.append(layer.state_dict()['weight'])
            self.bias.append(layer.state_dict()['bias'])

        # Parameter
        self.p = casadi.MX.sym("p", (self.state_dim, 1))
        self.h_current = casadi.MX.sym("hc")
        self.delta_h = casadi.MX.sym("dh")
        self.u_p = casadi.MX.sym("up", (self.input_dim, 1))

        # Optimization variable
        self.u = casadi.MX.sym("u", (self.input_dim, 1))
        self.x = casadi.MX(lin_sys.A @ self.p) + lin_sys.B @ self.u
        self.xi = casadi.MX.sym("xi")

        # NN Version A
        if m_type == LinModelType.A or m_type == LinModelType.APLUS:
            self.x1 = self.weights[0].numpy() @ self.x + \
                self.bias[0].numpy()  # First layer
            # self.x1out = fmax(self.x1,0) # relu
            self.x1out = casadi.log(1 + casadi.exp(self.x1))  # softplus
            # self.x1out = 1 / (1 + casadi.exp(-self.x1)) #sigmoid

            self.x2 = self.weights[1].numpy(
            ) @ self.x1out + self.bias[1].numpy()  # Second Layer
            # self.x2out = fmax(self.x2,0) # relu
            self.x2out = casadi.log(1 + casadi.exp(self.x2))  # softplus
            # self.x2out = 1 / (1 + casadi.exp(-self.x2)) #sigmoid

            self.fout1 = self.weights[2].numpy(
            ) @ self.x2out + self.bias[2].numpy()  # Third Layer

        if m_type == LinModelType.B or m_type == LinModelType.C or m_type == LinModelType.BPLUS or m_type == LinModelType.CPLUS:
            # NN Version B also C
            self.x1 = self.weights[0].numpy() @ self.x + \
                self.bias[0].numpy()  # First layer
            self.x1out = casadi.log(1 + casadi.exp(self.x1))  # softplus

            self.x2 = self.weights[1].numpy(
            ) @ self.x1out + self.bias[1].numpy()  # Second Layer
            self.x2out = casadi.log(1 + casadi.exp(self.x2))  # softplus

            self.x3 = self.weights[2].numpy(
            ) @ self.x2out + self.bias[2].numpy()  # Second Layer
            self.x3out = casadi.log(1 + casadi.exp(self.x3))  # softplus

            self.fout1 = self.weights[3].numpy(
            ) @ self.x3out + self.bias[3].numpy()  # Third Layer

        # Plus version have soft plus output
        if m_type == LinModelType.APLUS or m_type == LinModelType.BPLUS or m_type == LinModelType.CPLUS:
            self.fout = casadi.log(1 + casadi.exp(self.fout1))  # softplus
        else:
            self.fout = self.fout1

        if not self.verbose:
            self.opts = {'ipopt.print_level': 0,
                         'print_time': 0}  # disable output,
        else:
            self.opts = {}

        # Objective and constraint formulation
        self.alpha = 1e5 # Slack penalty
        self.f1 = self.fout
        self.f2 = (self.u_p - self.u).T @ (self.u_p -
                                           self.u) + self.alpha * self.xi

        self.g2_1 = self.fout - self.h_current - self.delta_h - self.zero_tol - self.xi
        self.g2_2 = -self.xi

        self.solver1 = casadi.nlpsol(
            "solver", "ipopt", {'x': self.u, 'p': self.p, 'f': self.f1}, self.opts)
        self.solver2 = casadi.nlpsol("solver", "ipopt", {'x': casadi.vertcat(self.u, self.xi), 'p': casadi.vertcat(
            self.p, self.h_current, self.delta_h, self.u_p), 'f': self.f2, 'g': casadi.vertcat(self.g2_1, self.g2_2)}, self.opts)

        self.xi_list = []
        self.hpb_traj = []
        self.min_decrease_list = []

    def input(self, x):
        """ Returns the next input"""
        # Obtain performance controller input
        u_p = self.performance_ctrl.input(x)
        # Evaluate approximate CBF at x(k)
        with torch.no_grad():
            h_x = self.model(torch.tensor(
                x).reshape(-1, self.state_dim).float()).numpy()
            # if self.use_log_space_model:
            #     h_x = output_modifier(h_x)

        self.hpb_traj.append(h_x[0])

        # 1. opt. problem - max decrease computaiton
        x = x.reshape((-1, 1))
        res1 = self.solver1(x0=0.0, p=x, lbx=self.lower_u, ubx=self.upper_u)
        delta_h = res1['f'].full()[0] - h_x[0]
        max_dec_u = res1['x'].full()[0]
        self.min_decrease_list.append(delta_h)

        # 2. opt. problem - safety filter using max decrease
        #init_u = [0.0,0.0]
        init_u = max_dec_u
        res2 = self.solver2(x0=init_u, p=casadi.vertcat(
            x, h_x, delta_h, u_p), lbx=self.lower_u, ubx=self.upper_u, ubg=[0.0])
        u = res2['x'].full()[:self.input_dim].flatten()
        #xi = res2['x'].full()[self.input_dim]

        # self.xi_list.append(xi)
        # if self.verbose :
        #print("succes ? ", self.solver.stats()['success'])
        return u
