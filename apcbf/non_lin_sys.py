""" Copyright (c) 2023, ETH Zurich, 
Alexandre Didier*, Robin C. Jacobs*, Jerome Sieber*, Kim P. Wabersich°, Prof. Dr. Melanie N. Zeilinger*, 
*Institute for Dynamic Systems and Control, D-MAVT
°Corporate Research of Robert Bosch GmbH
All rights reserved."""

""" This module contains the system dynamics definition for the nonlinear system used in this work"""
from apcbf.dynamic import *
import numpy as np

# Parameters taken from paper
vs = 5.0
L = 5.0
T_s = T_disc = 0.05

def xdot_nonlinear(t, x, u):
    target_vel = vs
    x_dot = np.zeros_like(x)
    x_dot[0] = (target_vel + x[3]) * np.sin(x[1])
    x_dot[1] = ((target_vel + x[3]) / L) * np.tan(x[2])
    x_dot[2] = u[0]
    x_dot[3] = u[1]
    return x_dot

non_lin_cont = NonLinearContinuousDynamics(xdot_nonlinear, 4, 2)

def disc_nonlinear(x, u):
    target_vel = vs
    x_next = np.zeros_like(x)
    x_next[0] = x[0] + T_disc * (target_vel + x[3]) * np.sin(x[1])
    x_next[1] = x[1] + T_disc * ((target_vel + x[3]) / L) * np.tan(x[2])
    x_next[2] = x[2] + T_disc * u[0]
    x_next[3] = x[3] + T_disc * u[1]
    return x_next

non_lin_disc = NonLinearDiscreteDynamics(disc_nonlinear, 4, 2)
