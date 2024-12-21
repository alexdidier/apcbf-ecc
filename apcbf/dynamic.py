""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

""" This module defines different types of dynamical systems"""
from abc import abstractmethod
import numpy as np


class DiscreteDynamics:
    """ Abstract class specifying discrete dynamical systems, with state x and input u"""
    @classmethod
    @abstractmethod
    def step(self, x, u):
        pass


class LinearDiscreteDynamics(DiscreteDynamics):
    """ This class defines a linear, discrete time dynamical system"""

    def __init__(self, A: np.mat, B: np.mat = np.mat(0)):
        super().__init__()
        self.A = A
        self.state_dim = A.shape[0]

        self.B = B
        self.input_dim = B.shape[1]

    def __str__(self) -> str:
        return f"LinearDiscreteDynamics(\n A: {self.A}, \n B: {self.B})"

    def step(self, x, u=0) -> np.array:
        return self.A @ x + (self.B @ u).reshape(self.state_dim, 1)


class NonLinearDiscreteDynamics(DiscreteDynamics):
    """ This class defines a nonlinear, discrete time dynamical system"""

    def __init__(self, next_state_function, state_dim, input_dim=0):
        super().__init__()
        self.function = next_state_function
        self.state_dim = state_dim
        self.input_dim = input_dim

    def __str__(self) -> str:
        return f"NonlinearDiscreteDynamics(\n f: {str(self.function.__str__)}"

    def step(self, x, u=0) -> np.array:
        return self.function(x, u).reshape(self.state_dim, 1)


class ContinuousDynamics:
    """ Abstract class specifying continuous dynamical systems, with state x, time t, and input u"""
    @classmethod
    @abstractmethod
    def xdot(self, t, x, u):
        pass

    @classmethod
    @abstractmethod
    def xdot_extended(self, t, x_extended):
        pass


class LinearContinuousDynamics(ContinuousDynamics):
    """ This class defines a linear, continuous time dynamical system"""

    def __init__(self, A: np.mat, B: np.mat = np.mat(0)):
        super().__init__()
        self.A = A
        self.state_dim = A.shape[0]
        self.B = B
        self.input_dim = B.shape[1]

    def __str__(self) -> str:
        return f"LinearContinuousDynamics (\n A: {self.A}, \n B: {self.B})"

    def xdot(self, t, x, u):
        x = x.reshape(-1, 1)
        u = u.reshape(-1, 1)
        return self.A @ x + (self.B @ u).reshape(self.state_dim, 1)

    def xdot_extended(self, t, x_extended):
        x = x_extended[:self.state_dim]
        u = x_extended[self.state_dim:]

        assert u.shape[0] == self.input_dim
        assert x.shape[0] == self.state_dim

        x_dot = self.xdot(t, x, u)
        u_dot = np.zeros_like(u).reshape(-1, 1)
        x_ext_dot = np.concatenate((x_dot, u_dot))

        return x_ext_dot.flatten()


class NonLinearContinuousDynamics(ContinuousDynamics):
    """ This class defines a nonlinear, continuous time dynamical system"""

    def __init__(self, xdot_t_x_u, state_dim, input_dim):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.function = xdot_t_x_u

    def __str__(self) -> str:
        return f"NonlinearContinuousDynamics(\n f: {str(self.function.__str__)}"

    def xdot(self, t, x, u):
        return self.function(t, x, u)

    def xdot_extended(self, t, x_extended):
        x = x_extended[:self.state_dim]
        u = x_extended[self.state_dim:]

        assert u.shape[0] == self.input_dim
        assert x.shape[0] == self.state_dim

        x_dot = self.xdot(t, x, u)
        u_dot = np.zeros_like(u)
        x_ext_dot = np.concatenate((x_dot, u_dot))

        return x_ext_dot.flatten()
